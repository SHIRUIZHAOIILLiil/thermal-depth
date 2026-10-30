"""Frozen DA2 + frozen caption-trained Iris; only spatial fusion modules learn."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


def id_noise(latent, sample_ids):
    if len(sample_ids) != len(latent):
        raise ValueError('One stable frame ID is required for each image')
    values = []
    for z, sid in zip(latent, sample_ids):
        seed = int.from_bytes(hashlib.sha256(('iris-fusion:43:' + sid).encode()).digest()[:8], 'little') % (2**63-1)
        generator = torch.Generator(device=z.device).manual_seed(seed)
        values.append(torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype))
    return torch.stack(values)


def padded_coordinates(x, image_hw, padded_hw, target_hw):
    """Half-pixel sampling: target covers DA2 padded image, source covers native image.

    Reflect beyond the native image boundary just as DA2's image padding does.
    Unlike resize(x, target_hw), this does not stretch native features into padding.
    """
    h, w = image_hw; hp, wp = padded_hw; th, tw = target_hw
    if hp < h or wp < w:
        raise ValueError('Invalid padded extent')
    def coordinates(native, padded, target):
        p = ((torch.arange(target, device=x.device, dtype=torch.float32) + .5) * padded / target - .5).clamp_min(0)
        if padded > native:
            # Image reflect-padding reflects about the last pixel CENTER,
            # whereas grid_sample reflection reflects about the outer edge.
            p = p.remainder(2*(native-1))
            p = torch.where(p > native-1, 2*(native-1)-p, p)
        else:
            p = p.clamp_max(native-1)
        return (p+.5)/native*2-1
    y = coordinates(h, hp, th)
    z = coordinates(w, wp, tw)
    yy, xx = torch.meshgrid(y, z, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None].expand(len(x), -1, -1, -1)
    return F.grid_sample(x.float(), grid, mode='bilinear', padding_mode='border', align_corners=False).to(x.dtype)


class SpatialFusion(nn.Module):
    def __init__(self, source_channels, width=128):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(source_channels, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(2*width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 1))
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)

    def forward(self, visual, source, image_hw, padded_hw):
        source = padded_coordinates(self.project(source), image_hw, padded_hw, visual.shape[-2:])
        return visual + self.merge(torch.cat((visual, source), 1))


class IrisFeatures(nn.Module):
    def __init__(self, unet, vae, text_encoder, channels=(640, 320)):
        super().__init__()
        self.unet = unet.requires_grad_(False).eval()
        self.vae = vae.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.channels = channels
        self._capture = None
        taps = [unet.up_blocks[2].attentions[2], unet.up_blocks[3].attentions[2]]
        self._handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(taps)]
        self.last_noise_hash = None

    def _hook(self, i):
        def hook(module, inputs, output):
            if self._capture is not None:
                self._capture[i] = output if torch.is_tensor(output) else output[0]
        return hook

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image, token_ids, mask, sample_ids, trace=False):
        if token_ids.shape != (len(image), 77) or mask.shape != token_ids.shape:
            raise ValueError('Both arms require exactly 77 tokens')
        with torch.autocast(device_type=image.device.type, enabled=False):
            latent = self.vae.encode(image.float()).latent_dist.mode() * self.vae.config.scaling_factor
            noise = id_noise(latent, sample_ids)
            # Preserve Iris causal CLIP / unmasked cross-attention. Mask is metadata.
            text = self.text_encoder(input_ids=token_ids).last_hidden_state
        self.last_noise_hash = hashlib.sha256(noise.cpu().contiguous().numpy().tobytes()).hexdigest() if trace else None
        task = torch.tensor([[1., 0.]], device=image.device).expand(len(image), -1)
        task = torch.cat((task.sin(), task.cos()), -1)
        self._capture = [None, None]
        try:
            self.unet(torch.cat((latent, noise), 1), 999, encoder_hidden_states=text,
                      class_labels=task, return_dict=False)
            result = self._capture
            for f, c, scale in zip(result, self.channels, (16, 8)):
                expected = (len(image), c, image.shape[-2]//scale, image.shape[-1]//scale)
                if f is None or tuple(f.shape) != expected or not torch.isfinite(f).all():
                    raise ValueError(f'Iris feature shape/finite failure: expected {expected}')
            return tuple(f.detach() for f in result)
        finally:
            self._capture = None


class FusionDepth(nn.Module):
    def __init__(self, core, iris, seed=43, source_channels=(640, 320), width=128):
        super().__init__()
        self.core = core.requires_grad_(False).eval()
        self.iris = iris.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.fusions = nn.ModuleList([SpatialFusion(c, width) for c in source_channels])
        self._context = None
        scratch = self.core.depth_head.scratch
        self._handles = [m.register_forward_hook(self._inject(i)) for i, m in
                         enumerate((scratch.layer3_rn, scratch.layer2_rn))]

    def _inject(self, i):
        def hook(module, inputs, output):
            if self._context is None:
                return output
            features, hw, padded_hw = self._context
            if output.shape[1] != self.fusions[i].merge[-1].out_channels:
                raise ValueError('Unexpected DPT scratch width')
            return self.fusions[i](output, features[i], hw, padded_hw)
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.core.eval(); self.iris.eval()
        return self

    @staticmethod
    def pad(image):
        h, w = image.shape[-2:]
        padding = (0, (-w) % 14, 0, (-h) % 14)
        return F.pad(image, padding, mode='reflect') if any(padding) else image

    def baseline(self, image):
        if self._context is not None:
            raise RuntimeError('Nested forward is not supported')
        h, w = image.shape[-2:]
        return self.core(self.pad(image))[..., :h, :w]

    def forward(self, image, iris_image, token_ids, attention_mask, sample_ids, trace=False):
        if image.shape != iris_image.shape:
            raise ValueError('The branches must use the same native image geometry')
        if self._context is not None:
            raise RuntimeError('Nested forward is not supported')
        features = self.iris(iris_image, token_ids, attention_mask, sample_ids, trace=trace)
        x = self.pad(image); h, w = image.shape[-2:]
        self._context = (features, (h, w), x.shape[-2:])
        try:
            # Do NOT use no_grad here: frozen DPT must carry gradients to fusion.
            return self.core(x)[..., :h, :w]
        finally:
            self._context = None


def load_iris(path, checkpoint):
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (payload.get('route') != 'b_thermal_unet' or payload.get('global_step') != 18000
            or payload.get('caption_mode') != 'correct' or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError('Expected converted caption Iris-G step18000 (not a dual/TADP/legacy checkpoint)')
    config = UNet2DConditionModel.load_config(path, subfolder='unet', local_files_only=True)
    config.update(in_channels=8, class_embed_type='projection', projection_class_embeddings_input_dim=4)
    unet = UNet2DConditionModel.from_config(config)
    unet.load_state_dict(payload['state_dicts']['unet'], strict=True)
    if unet.config.cross_attention_dim != 1024:
        raise ValueError('Expected SD2.1 cross attention')
    del payload
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', local_files_only=True, use_safetensors=True)
    text = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', local_files_only=True)
    return IrisFeatures(unet, vae, text)
