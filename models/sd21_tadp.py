"""TADP-style direct metric depth on SD2.1 (not a Lotus latent-depth model).

Feature locations match TADP's LDM output_blocks [1, 4, 7, 11], BEFORE
upsampling: resnet 1 of up block 0, attention 1/1/2 of up blocks 1/2/3.
No downloaded TADP code, attention replacement, task switcher or GT encoding.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DepthHead(nn.Module):
    def __init__(self, channels=(320, 640, 1280, 1280), width=1536, filters=32, out=192):
        super().__init__()
        a, b, c, d = channels
        self.channels = tuple(channels)
        self.layer1 = nn.Sequential(nn.Conv2d(a, a, 3, 2, 1), nn.GroupNorm(16, a),
                                    nn.ReLU(), nn.Conv2d(a, a, 3, 2, 1))
        self.layer2 = nn.Conv2d(b, b, 3, 2, 1)
        self.aggregate = nn.Sequential(nn.Conv2d(a+b+c+d, width, 1),
                                       nn.GroupNorm(16, width), nn.ReLU())
        layers = []
        for i in range(3):
            layers += [nn.ConvTranspose2d(width if i == 0 else filters, filters, 2, 2, bias=False),
                       nn.BatchNorm2d(filters), nn.ReLU()]
        self.decoder = nn.Sequential(*layers, nn.Conv2d(filters, out, 3, padding=1),
                                     nn.BatchNorm2d(out), nn.ReLU())
        self.depth = nn.Sequential(nn.Conv2d(out, out, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(out, 1, 3, padding=1))
        for m in (self.layer1, self.layer2, self.aggregate):
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.trunc_normal_(layer.weight, std=.02)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)
        for m in (self.decoder, self.depth):
            for layer in m.modules():
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.normal_(layer.weight, std=.001)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, features, size):
        a, b, c, d = features
        expected = [(size[0]//s, size[1]//s) for s in (8, 16, 32, 64)]
        for f, ch, hw in zip(features, self.channels, expected):
            if f.shape[1] != ch or tuple(f.shape[-2:]) != hw:
                raise ValueError(f'Wrong feature tap: {tuple(f.shape)}; expected {ch}, {hw}')
        d = F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False)
        x = self.aggregate(torch.cat((self.layer1(a), self.layer2(b), c, d), dim=1))
        x = self.decoder(x)
        # Two x2 bilinear stages, as in the TADP depth decoder.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Keep sigmoid/log-loss inputs FP32 even under autocast.
        return (80. * torch.sigmoid(self.depth(x).float())).squeeze(1)


class SD21Depth(nn.Module):
    def __init__(self, unet, vae, text_encoder, head=None):
        super().__init__()
        self.unet = unet.requires_grad_(True)
        self.vae = vae.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.head = head if head is not None else DepthHead()
        # Final denoising projection is unused; do not optimize it.
        self.unet.conv_norm_out = None
        self.unet.conv_act = None
        self.unet.conv_out = nn.Identity()
        if hasattr(self.vae, 'decoder'): self.vae.decoder = nn.Identity()
        if hasattr(self.vae, 'post_quant_conv'): self.vae.post_quant_conv = nn.Identity()
        self._capture = None
        taps = [unet.up_blocks[0].resnets[1], unet.up_blocks[1].attentions[1],
                unet.up_blocks[2].attentions[1], unet.up_blocks[3].attentions[2]]
        self._handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(taps)]

    def _hook(self, index):
        def capture(module, inputs, output):
            # Checkpoint recomputation during backward must not retain stale graphs.
            if self._capture is not None:
                self._capture[index] = output if torch.is_tensor(output) else (
                    output[0] if isinstance(output, tuple) else output.sample)
        return capture

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval(); self.text_encoder.eval()
        return self

    def forward(self, image, token_ids, attention_mask=None):
        h, w = image.shape[-2:]
        if h % 64 or w % 64:
            raise ValueError('Native thermal input must be divisible by 64; never resize sparse GT')
        if token_ids.shape != (image.shape[0], 77): raise ValueError('Expected B x 77 token IDs')
        with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            latent = self.vae.encode(image.float()).latent_dist.mode() * self.vae.config.scaling_factor
            # SD2 native convention: causal CLIP, no explicit padding mask.
            text = self.text_encoder(input_ids=token_ids).last_hidden_state
        self._capture = [None] * 4
        try:
            self.unet(latent, torch.ones(image.shape[0], device=image.device, dtype=torch.long),
                      encoder_hidden_states=text, return_dict=False)
            features = self._capture[::-1]
            if any(f is None for f in features): raise RuntimeError('Missing SD2 feature tap')
        finally:
            self._capture = None
        return self.head(features, (h, w))


def load_sd21(path):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel
    unet = UNet2DConditionModel.from_pretrained(path, subfolder='unet', local_files_only=True,
                                               use_safetensors=True)
    c = unet.config
    if (c.in_channels != 4 or c.cross_attention_dim != 1024 or
            tuple(c.block_out_channels) != (320, 640, 1280, 1280) or
            not c.use_linear_projection or c.class_embed_type is not None or c.sample_size != 64):
        raise ValueError('Requires original SD2.1-base, not Lotus-G/D or SD1.5')
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', local_files_only=True, use_safetensors=True)
    text = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', local_files_only=True)
    if text.config.hidden_size != 1024: raise ValueError('Wrong text encoder')
    unet.enable_gradient_checkpointing()
    return SD21Depth(unet, vae, text)
