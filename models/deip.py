"""Differentiable Imaging Physics-driven Data Prior Enhancement (DEIP, Eq.2-9).

The DEIP constraint parameters (rho_c, rho_h, s, omega, sigma_att, mu) are
learnable, and Gumbel-Softmax keeps the discrete high-frequency sampling
differentiable during training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p):
    """Inverse sigmoid used to initialize the probability-valued parameters."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class DEIPModule(nn.Module):
    """Differentiable DEIP enhancement (Eq.2-9) operating on image batches."""

    def __init__(self, init_rho_c=0.4, init_rho_h=0.5, init_s=1.0, init_omega=5.0,
                 init_sigma_att=0.08, init_mu=30.0, smoothing_sigma=3.0):
        """Create the learnable DEIP parameters with the given initial values."""
        super().__init__()
        self.logit_rho_c = nn.Parameter(torch.tensor(_logit(init_rho_c)))
        self.logit_rho_h = nn.Parameter(torch.tensor(_logit(init_rho_h)))
        self.log_s = nn.Parameter(torch.tensor(math.log(max(float(init_s), 1e-6))))
        self.log_omega = nn.Parameter(torch.tensor(math.log(max(float(init_omega), 1e-6))))
        self.log_sigma_att = nn.Parameter(torch.tensor(math.log(max(float(init_sigma_att), 1e-6))))
        self.log_mu = nn.Parameter(torch.tensor(math.log(max(float(init_mu), 1e-6))))
        self.smoothing_sigma = float(smoothing_sigma)
        self._grid_cache = {}

    @property
    def rho_c(self):
        return torch.sigmoid(self.logit_rho_c)

    @property
    def rho_h(self):
        return torch.sigmoid(self.logit_rho_h)

    @property
    def s(self):
        return torch.exp(self.log_s)

    @property
    def omega(self):
        return torch.exp(self.log_omega)

    @property
    def sigma_att(self):
        return torch.exp(self.log_sigma_att)

    @property
    def mu(self):
        return torch.exp(self.log_mu)

    def _build_grids(self, h, w, device):
        """Build and cache the frequency and spatial coordinate grids."""
        key = (h, w)
        if key in self._grid_cache:
            return self._grid_cache[key]

        v_grid = torch.linspace(0.0, math.pi, h, device=device).view(h, 1)

        u = torch.linspace(-1.0, 1.0, w, device=device).view(1, w)
        v = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1)
        r_grid = torch.sqrt(u ** 2 + v ** 2) / math.sqrt(2.0)

        gy, gx = torch.meshgrid(torch.linspace(-1.0, 1.0, h, device=device),
                                torch.linspace(-1.0, 1.0, w, device=device),
                                indexing='ij')
        coords = torch.stack([gx, gy], dim=-1).view(1, h, w, 2)

        self._grid_cache[key] = (v_grid, r_grid, coords)
        return v_grid, r_grid, coords

    def _gaussian_kernel(self, sigma, device):
        """Return a normalized 2D Gaussian smoothing kernel."""
        ks = int(2 * math.ceil(3 * sigma) + 1)
        center = ks // 2
        t = torch.arange(ks, device=device).float() - center
        g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.view(1, 1, 1, ks) * g.view(1, 1, ks, 1)
        return kernel

    @torch.no_grad()
    def _intensity_centroid(self, gray):
        """Estimate a lesion center from the centroid of the salient region.

        Returns the normalized [B, 2] (cx, cy) coordinates used to center the
        DEIP deformation field.
        """
        x = gray.squeeze(1)
        thr = x.view(x.size(0), -1).max(dim=1, keepdim=True).values * 0.5
        mask = (x > thr.unsqueeze(-1)).float()
        h, w = x.shape[1], x.shape[2]
        gy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, h, 1)
        gx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, w)
        total = mask.sum(dim=(1, 2)).clamp(min=1)
        cx = (mask * gx).sum(dim=(1, 2)) / total
        cy = (mask * gy).sum(dim=(1, 2)) / total
        return torch.stack([cx, cy], dim=1)

    def forward(self, x, lesion_center=None):
        """Apply the DEIP enhancement (Eq.2-9) to a batch of normalized images.

        The images are Fourier transformed (Eq.2), undersampled with a mask built
        from the rho_c center band and a straight-through Gumbel Bernoulli(rho_h)
        draw (Eq.3-4), modulated by the sinusoidal motion phase (Eq.5), and
        reconstructed in the magnitude domain (Eq.6). A lesion-centered Gaussian
        field and a smoothed random displacement then warp the image (Eq.7-9).
        Takes [B, 3, H, W] inputs and an optional [B, 2] lesion center, and
        returns [B, 3, H, W] augmented images.
        """
        b, c, h, w = x.shape
        device = x.device
        gray = x.mean(dim=1, keepdim=True)

        v_grid, r_grid, coords = self._build_grids(h, w, device)

        K = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))

        rho_c = self.rho_c
        tau = 0.1
        center_mask = torch.sigmoid((rho_c - r_grid) / tau)
        logits = torch.stack([self.logit_rho_h.expand(b, h, w),
                              -self.logit_rho_h.expand(b, h, w)], dim=-1)
        gumbel = F.gumbel_softmax(logits, hard=True, dim=-1)
        high_freq_mask = gumbel[..., 0]
        M = center_mask + (1 - center_mask) * high_freq_mask
        K_prime = M.unsqueeze(1) * K

        phase = torch.exp(1j * self.s * torch.sin(self.omega * v_grid))
        K_double = K_prime * phase.view(1, 1, h, 1)

        I_prime = torch.fft.ifft2(torch.fft.ifftshift(K_double, dim=(-2, -1))).abs()

        if lesion_center is None:
            lesion_center = self._intensity_centroid(gray)
        cx = lesion_center[:, 0:1, None, None]
        cy = lesion_center[:, 1:2, None, None]
        gx = coords[..., 0].unsqueeze(0)
        gy = coords[..., 1].unsqueeze(0)
        dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
        weight = torch.exp(-dist2 / (2 * self.sigma_att ** 2))

        rand = torch.rand(b, 1, h, w, device=device) * 2 - 1
        kernel = self._gaussian_kernel(self.smoothing_sigma, device)
        phi_x = F.conv2d(rand, kernel, padding=kernel.shape[-1] // 2)
        phi_y = F.conv2d(rand, kernel, padding=kernel.shape[-1] // 2)

        dx = self.mu * weight * phi_x * (2.0 / w)
        dy = self.mu * weight * phi_y * (2.0 / h)
        offsets = torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)
        grid = coords + offsets

        aug = F.grid_sample(I_prime, grid, mode='bilinear',
                            padding_mode='reflection', align_corners=False)
        return aug.expand(-1, 3, -1, -1)
