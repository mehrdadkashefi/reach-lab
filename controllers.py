"""
controllers.py -- controllers for the arm26 reaching task.

GRUController : single-population GRU (the baseline).
ModularGRU   : three GRU populations (e.g. PMd / M1 / spinal cord) implemented as one fused
               GRU over all units, with fixed probabilistic masks on the input, recurrent,
               and readout weights. Masked synapses stay at zero and receive no gradient.

Both expose the same interface:
    model.hidden_dim
    model.start_sequence()           # call once before each rollout (no-op for the baseline)
    out, h = model(x, h)             # out in [0,1], shape (batch, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUController(nn.Module):
    """Single GRU population."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=6, out_bias=-2.0):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        nn.init.constant_(self.fc.bias, out_bias)

    def start_sequence(self):
        pass

    def forward(self, x, h):
        h = self.gru(x, h)
        return torch.sigmoid(self.fc(h)), h


class ModularGRU(nn.Module):
    """Three GRU modules with probabilistic, masked connectivity, as one fused GRU.

    All architecture parameters (module sizes, the input-stream masks, the module-to-module
    connectivity, the output mask, and the spectral scaling) have sensible defaults here and
    are only changed if passed as kwargs.

    Args:
        input_dim:    number of input channels.
        output_dim:   number of actuator channels.
        input_layout: list of (stream_type, n_channels) describing the input vector, where
                      stream_type is one of 'task' / 'vision' / 'proprio'. The matching
                      *_mask below sets P(channel in that stream -> each module).
        module_sizes: module sizes, order [PMd, M1, spinal].
        vision_mask / proprio_mask / task_mask: length-n_modules P(stream -> each module).
        output_mask:  length-n_modules P(unit in module -> output).
        connectivity: (n_modules x n_modules) matrix; entry [i,j] = P(module j -> module i).
        spectral_scaling: target spectral radius of the masked candidate recurrent block.
        seed:         RNG seed for the (fixed) connectivity masks.
    """
    DEFAULT_CONNECTIVITY = [[1.0, 0.2, 0.02],
                            [0.2, 1.0, 0.2],
                            [0.0, 0.2, 1.0]]

    def __init__(self, input_dim, output_dim, input_layout,
                 module_sizes=(256, 256, 32),
                 vision_mask=(0.2, 0.0, 0.0),
                 proprio_mask=(0.0, 0.0, 0.5),
                 task_mask=(0.2, 0.02, 0.0),
                 output_mask=(0.0, 0.0, 0.5),
                 connectivity=None,
                 spectral_scaling=1.1,
                 out_bias=-2.0, seed=0):
        super().__init__()
        if connectivity is None:
            connectivity = self.DEFAULT_CONNECTIVITY
        # build input streams (column ranges + per-stream module masks) from the layout
        stream_masks = {'vision': vision_mask, 'proprio': proprio_mask, 'task': task_mask}
        input_streams, c = [], 0
        for stype, nch in input_layout:
            input_streams.append((c, c + nch, stream_masks[stype])); c += nch

        g = torch.Generator().manual_seed(seed)
        H = int(sum(module_sizes))
        self.hidden_dim = H
        self.H = H

        mod_id = torch.cat([torch.full((int(sz),), m, dtype=torch.long)
                            for m, sz in enumerate(module_sizes)])              # (H,)
        connectivity = torch.as_tensor(connectivity, dtype=torch.float32)
        output_mask = torch.as_tensor(output_mask, dtype=torch.float32)

        # recurrent mask: P(unit_j -> unit_i) = connectivity[mod_i, mod_j]
        prob_hh = connectivity[mod_id][:, mod_id]                               # (H,H)
        hh_mask = (torch.rand(H, H, generator=g) < prob_hh).float()

        # input mask: per stream, P(channel -> module)
        prob_ih = torch.zeros(H, input_dim)
        for (c0, c1, stream_mask) in input_streams:
            sm = torch.as_tensor(stream_mask, dtype=torch.float32)
            prob_ih[:, c0:c1] = sm[mod_id].unsqueeze(1)
        ih_mask = (torch.rand(H, input_dim, generator=g) < prob_ih).float()

        # output mask: P(unit -> output) = output_mask[mod_i]
        prob_out = output_mask[mod_id].unsqueeze(0).expand(output_dim, H)
        out_mask = (torch.rand(output_dim, H, generator=g) < prob_out).float()

        # fused GRU parameters (gates reset r, update z, candidate n stacked along dim 0)
        self.weight_ih = nn.Parameter(torch.empty(3 * H, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * H, H))
        self.bias_ih = nn.Parameter(torch.zeros(3 * H))
        self.bias_hh = nn.Parameter(torch.zeros(3 * H))
        self.readout = nn.Linear(H, output_dim)
        nn.init.constant_(self.readout.bias, out_bias)

        std = 1.0 / (H ** 0.5)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)

        # spectral scaling of the masked candidate recurrent block (rows 2H:3H = W_hn)
        with torch.no_grad():
            whn = (self.weight_hh[2 * H:3 * H, :] * hh_mask).cpu()
            radius = torch.linalg.eigvals(whn).abs().max().item()
            if radius > 0:
                self.weight_hh[2 * H:3 * H, :].mul_(spectral_scaling / radius)

        # tile masks across the 3 gates; register as buffers so .to(device) moves them
        self.register_buffer('ih_mask', ih_mask.repeat(3, 1))                  # (3H, input_dim)
        self.register_buffer('hh_mask', hh_mask.repeat(3, 1))                  # (3H, H)
        self.register_buffer('out_mask', out_mask)                             # (output_dim, H)
        self.register_buffer('mod_id', mod_id)

        self._Wih = self._Whh = self._Wout = None

    def density(self):
        """Realized connection fraction of each mask (for reporting)."""
        return (self.ih_mask.mean().item(), self.hh_mask.mean().item(), self.out_mask.mean().item())

    def start_sequence(self):
        # masked effective weights computed once per rollout (kept in the graph for backprop)
        self._Wih = self.weight_ih * self.ih_mask
        self._Whh = self.weight_hh * self.hh_mask
        self._Wout = self.readout.weight * self.out_mask

    def forward(self, x, h):
        if self._Wih is None:
            self.start_sequence()
        gi = F.linear(x, self._Wih, self.bias_ih)
        gh = F.linear(h, self._Whh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1.0 - z) * n + z * h
        out = F.linear(h, self._Wout, self.readout.bias)
        return torch.sigmoid(out), h