import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# -----------------------------------------------------------------------------
# Mamba selective scan import
# -----------------------------------------------------------------------------
MAMBA_AVAILABLE = False
MAMBA_IMPORT_ERROR = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
    print("mamba_ssm found, P_SSM will use the official CUDA kernel for acceleration.")
except Exception as e:
    MAMBA_AVAILABLE = False
    MAMBA_IMPORT_ERROR = e
    selective_scan_fn = None
    print(f"mamba_ssm import failed, P_SSM will use the reference Python implementation (selective_scan_ref). Error: {e}")


def selective_scan_ref(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B = B.float()
    C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])

        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)

    y = torch.stack(ys, dim=2)  # [B, D, L]
    out = y if D is None else y + u * D.view(1, -1, 1)

    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class P_SSM(nn.Module):
    """
    Paper-aligned DFM style cross fusion module.
    Input:
        Ff: [B, C, H, W]   # PCM * FAM output
        Fs: [B, C, H, W]   # spatial / deep feature
    Output:
        fused: [B, C, H, W]
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # -------- branch norms --------
        self.norm_ssm = nn.LayerNorm(d_model)
        self.norm_gate = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # -------- input / output projection --------
        self.in_proj = nn.Conv1d(d_model, self.d_inner, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv1d(self.d_inner, d_model, kernel_size=1, bias=bias)

        # -------- gate branch: MLP + activation -> weights Z --------
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, 2, bias=True)   # two-direction weights
        )

        # -------- output MLP --------
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True)
        )

        # -------- merge after de-interleave --------
        self.merge = nn.Conv2d(2 * d_model, d_model, kernel_size=1, bias=False)

        # -------- SSM params (2 directions) --------
        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                **factory_kwargs
            ),
            self.dt_init(
                self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                **factory_kwargs
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

        self.selective_scan = selective_scan_fn if MAMBA_AVAILABLE else selective_scan_ref

    @staticmethod
    def dt_init(
        dt_rank, d_inner, dt_scale=1.0, dt_init="random",
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
        **factory_kwargs
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_seq(self, x_seq: torch.Tensor):
        """
        x_seq: [B, d_inner, Lmix]
        return:
            y_fwd, y_bwd: [B, d_inner, Lmix]
        """
        B, D, Lmix = x_seq.shape
        K = 2

        xs = torch.stack([x_seq, torch.flip(x_seq, dims=[-1])], dim=1)  # [B, 2, D, Lmix]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, Lmix)                # [B, 2D, Lmix]
        dts = dts.contiguous().float().view(B, -1, Lmix) # [B, 2D, Lmix]
        Bs = Bs.float().view(B, K, -1, Lmix)
        Cs = Cs.float().view(B, K, -1, Lmix)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, Lmix)

        y_fwd = out_y[:, 0]
        y_bwd = torch.flip(out_y[:, 1], dims=[-1])

        return y_fwd, y_bwd

    def forward(self, Ff: torch.Tensor, Fs: torch.Tensor, **kwargs):
        """
        Ff, Fs: [B, C, H, W]
        """
        assert Ff.shape == Fs.shape, f"Shape mismatch: {Ff.shape} vs {Fs.shape}"

        B, C, H, W = Ff.shape
        L = H * W

        # 1) cross-mix at the same spatial location
        s1 = Ff.reshape(B, C, L)
        s2 = Fs.reshape(B, C, L)
        s_mix = torch.stack((s1, s2), dim=-1).reshape(B, C, 2 * L)  # [B, C, 2L]

        # 2) gate branch -> weights Z
        gate_tokens = self.norm_gate(s_mix.permute(0, 2, 1))  # [B, 2L, C]
        z_logits = self.gate_mlp(gate_tokens)                 # [B, 2L, 2]
        z = torch.softmax(z_logits, dim=-1)
        z_fwd = z[..., 0].unsqueeze(1)                        # [B, 1, 2L]
        z_bwd = z[..., 1].unsqueeze(1)                        # [B, 1, 2L]

        # 3) SSM branch -> bidirectional modeling
        ssm_tokens = self.norm_ssm(s_mix.permute(0, 2, 1)).permute(0, 2, 1)  # [B, C, 2L]
        ssm_tokens = self.in_proj(ssm_tokens)                                  # [B, d_inner, 2L]

        y_fwd, y_bwd = self.forward_core_seq(ssm_tokens)

        # 4) weighted fusion of bidirectional outputs
        y = z_fwd * y_fwd + z_bwd * y_bwd

        # 5) output projection + MLP + residual
        y = self.out_proj(y)                            # [B, C, 2L]
        y_tokens = self.norm_out(y.permute(0, 2, 1))   # [B, 2L, C]
        y_tokens = self.out_mlp(y_tokens)
        y = y_tokens.permute(0, 2, 1) + s_mix          # residual on mixed sequence

        # 6) de-interleave back to two streams
        out_f = y[..., ::2].reshape(B, C, H, W)
        out_s = y[..., 1::2].reshape(B, C, H, W)

        # 7) merge to one fused feature
        fused = self.merge(torch.cat([out_f, out_s], dim=1))
        fused = fused + 0.5 * (Ff + Fs)

        return fused