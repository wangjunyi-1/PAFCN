import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
import kornia.filters as KFilters

from einops import rearrange
from .SSM import P_SSM


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res, ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2] * ade
        x = self.post_conv(x)
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2
            )
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            bias=False
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2
            ),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = F.relu(self.weights)
        fuse_weights = weights / (weights.sum(dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x



class GradientCovarianceLayer(nn.Module):
    """


    输出:
        A_obj: [B, C, H, W]
    若 return_aux=True:
        A_obj, lambda1, lambda2, T
    """

    def __init__(
            self,
            kernel_size=5,
            eps=1e-6,
            init_tau=0.03,
            init_alpha=9.0,
            return_aux=False
    ):
        super().__init__()
        self.eps = eps
        self.return_aux = return_aux
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

        init_alpha = float(init_alpha)
        raw_alpha = torch.log(torch.exp(torch.tensor(init_alpha)) - 1.0)
        self.alpha_raw = nn.Parameter(raw_alpha)

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw) + self.eps

    def forward(self, x):
        squeeze_back = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D input, got {tuple(x.shape)}")

        x = x.float()

        grads = KFilters.spatial_gradient(x, order=1, mode="sobel", normalized=True)
        gy = grads[:, :, 0, :, :]
        gx = grads[:, :, 1, :, :]

        Jxx = self.pool(gx * gx)
        Jxy = self.pool(gx * gy)
        Jyy = self.pool(gy * gy)

        trace = Jxx + Jyy
        delta = torch.sqrt(torch.clamp((Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2), min=self.eps))

        lambda1 = 0.5 * (trace + delta)
        lambda2 = 0.5 * (trace - delta)

        lambda1 = torch.clamp(lambda1, min=self.eps)
        lambda2 = torch.clamp(lambda2, min=0.0)

        T = lambda2 / (lambda1 + self.eps)

        A_obj = 1.0 - torch.sigmoid(self.alpha * (T - self.tau))

        if squeeze_back:
            A_obj = A_obj.squeeze(0)
            lambda1 = lambda1.squeeze(0)
            lambda2 = lambda2.squeeze(0)
            T = T.squeeze(0)

        if self.return_aux:
            return A_obj, lambda1, lambda2, T
        return A_obj



def patch_split(input_tensor, patch_size_h, patch_size_w):
    """
    (B, C, H, W) -> (B*num_patches, C, patch_h, patch_w)
    """
    B, C, H, W = input_tensor.size()
    num_patches_h = H // patch_size_h
    num_patches_w = W // patch_size_w

    out = input_tensor.view(B, C, num_patches_h, patch_size_h, num_patches_w, patch_size_w)
    out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_size_h, patch_size_w)
    return out


def patch_recover(input_tensor, num_patches_h, num_patches_w, H, W):
    """
    (B*num_patches, C, patch_h, patch_w) -> (B, C, H, W)
    """
    N, C, patch_h, patch_w = input_tensor.size()
    B = N // (num_patches_h * num_patches_w)

    out = input_tensor.view(B, num_patches_h, num_patches_w, C, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return out



class DeformableFeatureEnhancer(nn.Module):

    def __init__(
            self,
            dim,
            num_patches_h=4,
            num_patches_w=4,
            scale_min=0.5,
            scale_max=1.5,
            max_rotation_deg=30.0,
            max_offset_ratio=0.5
    ):
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.dim = dim

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_rotation = math.radians(max_rotation_deg)
        self.max_offset_ratio = max_offset_ratio

        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 5, kernel_size=1)  # [sx, sy, angle, tx, ty]
        )

        self.final_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.wf = WF(in_channels=dim, decode_channels=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        assert H % self.num_patches_h == 0 and W % self.num_patches_w == 0, \
            f"H={H}, W={W} must be divisible by num_patches_h={self.num_patches_h}, num_patches_w={self.num_patches_w}"

        patch_h = H // self.num_patches_h
        patch_w = W // self.num_patches_w


        patches = patch_split(x, patch_h, patch_w)
        num_patches_total = patches.shape[0]
        num_patches_per_image = self.num_patches_h * self.num_patches_w


        params = self.param_predictor(patches).squeeze(-1).squeeze(-1)
        raw_scales = params[:, 0:2]
        raw_angle = params[:, 2]
        raw_offsets = params[:, 3:5]


        scales = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(raw_scales)
        sx = scales[:, 0]
        sy = scales[:, 1]

        angle = torch.tanh(raw_angle) * self.max_rotation

        max_tx = self.max_offset_ratio * (patch_w / W)
        max_ty = self.max_offset_ratio * (patch_h / H)
        tx = torch.tanh(raw_offsets[:, 0]) * max_tx
        ty = torch.tanh(raw_offsets[:, 1]) * max_ty
        offsets = torch.stack([tx, ty], dim=-1)


        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, patch_h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, patch_w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        base_patch_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(num_patches_total, 1, 1, 1)

        x_local = base_patch_grid[..., 0]
        y_local = base_patch_grid[..., 1]


        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        x_aff = (
                sx[:, None, None] * x_local * cos_a[:, None, None]
                - sy[:, None, None] * y_local * sin_a[:, None, None]
        )
        y_aff = (
                sx[:, None, None] * x_local * sin_a[:, None, None]
                + sy[:, None, None] * y_local * cos_a[:, None, None]
        )

        transformed_patch_grid = torch.stack([x_aff, y_aff], dim=-1)


        ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / H) - 1.0
        xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / W) - 1.0
        global_grid_y_full, global_grid_x_full = torch.meshgrid(ys, xs, indexing='ij')
        global_grid = torch.stack([global_grid_x_full, global_grid_y_full], dim=-1)  # [H, W, 2]

        centers_pooled = F.avg_pool2d(
            global_grid.permute(2, 0, 1).unsqueeze(0),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        )  # [1, 2, num_patches_h, num_patches_w]

        centers_one_image = centers_pooled.squeeze(0).permute(1, 2, 0).reshape(
            num_patches_per_image, 1, 1, 2
        )
        global_patch_centers = centers_one_image.repeat(B, 1, 1, 1)


        scale_factor = torch.tensor(
            [patch_w / W, patch_h / H],
            device=x.device,
            dtype=x.dtype
        ).view(1, 1, 1, 2)

        final_sampling_grid = (
                global_patch_centers
                + offsets[:, None, None, :]
                + transformed_patch_grid * scale_factor
        )
        final_sampling_grid = final_sampling_grid.clamp(-0.999, 0.999)

        # [B*num_patches, patch_h, patch_w, 2] -> [B, H, W, 2]
        final_sampling_grid = final_sampling_grid.view(
            B, self.num_patches_h, self.num_patches_w, patch_h, patch_w, 2
        )
        final_sampling_grid = final_sampling_grid.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, 2)


        enhanced_features = F.grid_sample(
            x,
            final_sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        out = self.wf(self.final_conv(enhanced_features), shortcut)
        return out


class JGFM(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int = 7,
            pcm_kernel_size: int = 5,
            num_patches_h: int = 4,
            num_patches_w: int = 4,
            scale_min: float = 0.5,
            scale_max: float = 1.5,
            max_rotation_deg: float = 15.0,
            max_offset_ratio: float = 0.5,
            theta_max_deg: float = 90.0,
            lambda_min: float = 2.0,
            lambda_max: float = 12.0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size


        self.feature_enhancer = DeformableFeatureEnhancer(
            dim=channels,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            scale_min=scale_min,
            scale_max=scale_max,
            max_rotation_deg=max_rotation_deg,
            max_offset_ratio=max_offset_ratio
        )


        self.theta_max = math.radians(theta_max_deg)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.gabor_theta_raw = nn.Parameter(torch.zeros(channels))
        self.gabor_lambda_raw = nn.Parameter(torch.zeros(channels))
        self.gabor_psi_raw = nn.Parameter(torch.zeros(channels))

        # PCM
        self.gradient_analyzer = GradientCovarianceLayer(
            kernel_size=pcm_kernel_size,
            return_aux=True
        )

    def _get_kernel_params(self):
        # theta in [-theta_max, theta_max]
        theta = torch.tanh(self.gabor_theta_raw) * self.theta_max

        # lambda in [lambda_min, lambda_max]
        gabor_lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(self.gabor_lambda_raw)

        # psi in [-pi, pi]
        psi = math.pi * torch.tanh(self.gabor_psi_raw)

        return theta, gabor_lambda, psi

    def _apply_gabor_filters(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        theta, gabor_lambda, psi = self._get_kernel_params()

        half = (self.kernel_size - 1) / 2.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-half, half, self.kernel_size, device=device, dtype=dtype),
            torch.linspace(-half, half, self.kernel_size, device=device, dtype=dtype),
            indexing='ij'
        )

        kernels = []
        for i in range(C):
            x_prime = x_grid * torch.cos(theta[i]) + y_grid * torch.sin(theta[i])

            kernel = torch.cos(2.0 * math.pi * x_prime / gabor_lambda[i] + psi[i])

            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.abs().sum() + 1e-6)

            kernels.append(kernel)

        gabor_kernels = torch.stack(kernels, dim=0).unsqueeze(1)  # [C, 1, K, K]

        g_response = F.conv2d(
            x,
            gabor_kernels,
            padding=self.kernel_size // 2,
            groups=C
        )
        return g_response

    def forward(self, x: torch.Tensor):
        x_enhanced = self.feature_enhancer(x)
        fam_response = self._apply_gabor_filters(x_enhanced)

        pcm_weight, lambda1, lambda2, T = self.gradient_analyzer(x)

        freq_response = fam_response * pcm_weight

        return freq_response


class PAFCNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 patch_size=8,
                 num_classes=6,
                 use_aux_loss=True,
                 dim_scale=8):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim_scale = dim_scale

        self.backbone = timm.create_model(
            model_name=backbone_name,
            features_only=True,
            pretrained=pretrained,
            output_stride=32,
            out_indices=(0, 1, 2, 3)
        )

        self.conv2 = ConvBN(2 * decode_channels, decode_channels, kernel_size=1)  # 192 -> 96
        self.conv3 = ConvBN(4 * decode_channels, decode_channels, kernel_size=1)  # 384 -> 96
        self.conv4 = ConvBN(8 * decode_channels, decode_channels, kernel_size=1)  # 768 -> 96

        self.fuse43 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.fuse32 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.fuse21 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.base_refine = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.ahg_filter = JGFM(channels=decode_channels)
        self.cross_scan_pvss = P_SSM(d_model=decode_channels, expand=1)
        self.innov_proj = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.innov_scale = nn.Parameter(torch.tensor(0.0))

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)
        )

        self.innov_proj = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.innov_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask=None, imagename=None):
        _, H, W = x.size()[-3:]

        res1, res2, res3, res4 = self.backbone(x)

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)

        r4_up = F.interpolate(res4, size=res3.shape[-2:], mode='bicubic', align_corners=False)
        p3 = self.fuse43(r4_up, res3)

        p3_up = F.interpolate(p3, size=res2.shape[-2:], mode='bicubic', align_corners=False)
        p2 = self.fuse32(p3_up, res2)

        p2_up = F.interpolate(p2, size=res1.shape[-2:], mode='bicubic', align_corners=False)
        base = self.fuse21(p2_up, res1)
        base = self.base_refine(base)

        g = self.ahg_filter(res1)

        innov = self.cross_scan_pvss(g, base)
        innov = self.innov_proj(innov)

        scale = torch.tanh(self.innov_scale)
        res = base + scale * innov

        res = self.segmentation_head(res)
        out = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
        return out