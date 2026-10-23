"""Frozen DA2 + frozen CLIP, trainable two-resolution log-depth correction."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Fusion(nn.Module):
    def __init__(self, vision_dim, text_dim, width=128):
        super().__init__()
        self.visual = nn.Linear(vision_dim, width)
        self.text = nn.Linear(text_dim, width)
        self.vnorm = nn.LayerNorm(width)
        self.tnorm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.merge = nn.Linear(2*width, width)

    def forward(self, visual, text, mask, patch_hw, target_hw):
        x = self.visual(visual).transpose(1, 2).reshape(visual.shape[0], -1, *patch_hw)
        x = F.interpolate(x, target_hw, mode='bilinear', align_corners=False)
        tokens = x.flatten(2).transpose(1, 2)
        text = self.tnorm(self.text(text))
        attended, _ = self.attention(self.vnorm(tokens), text, text, key_padding_mask=~mask.bool(), need_weights=False)
        return self.merge(torch.cat((tokens, attended), -1)).transpose(1, 2).reshape_as(x)


class ResidualHead(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024, width=128):
        super().__init__()
        self.fusions = nn.ModuleList([Fusion(vision_dim, text_dim, width) for _ in range(2)])
        self.combine = nn.Sequential(nn.Conv2d(2*width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.combine[-1].weight)
        nn.init.zeros_(self.combine[-1].bias)

    def forward(self, features, text, mask, patch_hw, image_hw):
        if mask.ndim != 2 or not mask.bool().any(dim=1).all():
            raise ValueError('Each caption needs at least one valid token')
        sizes = [tuple((v+d-1)//d for v in image_hw) for d in (8, 16)]
        fine, coarse = [f(x[0], text, mask, patch_hw, size)
                        for f, x, size in zip(self.fusions, features[2:], sizes)]
        coarse = F.interpolate(coarse, sizes[0], mode='bilinear', align_corners=False)
        u = self.combine(torch.cat((fine, coarse), 1))
        u = F.interpolate(u.float(), image_hw, mode='bilinear', align_corners=False).squeeze(1)
        return math.log(2.) * u.tanh()


class ResidualDepth(nn.Module):
    def __init__(self, core, text_encoder, seed=43, vision_dim=768, text_dim=1024, width=128):
        super().__init__()
        self.core = core.requires_grad_(False).eval()
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.residual = ResidualHead(vision_dim, text_dim, width)

    def train(self, mode=True):
        super().train(mode)
        self.core.eval(); self.text_encoder.eval()
        return self

    def forward(self, image, token_ids=None, attention_mask=None, details=False):
        if token_ids is None or attention_mask is None: raise ValueError('Both arms need text tokens/mask')
        h, w = image.shape[-2:]
        pad = (0, (-w)%14, 0, (-h)%14)
        x = F.pad(image, pad, mode='reflect') if any(pad) else image
        ph, pw = x.shape[-2]//14, x.shape[-1]//14
        with torch.no_grad():
            features = list(self.core.pretrained.get_intermediate_layers(
                x, self.core.intermediate_layer_idx[self.core.encoder], return_class_token=True))
            base = (self.core.depth_head(features, ph, pw).squeeze(1)*self.core.max_depth)[..., :h, :w]
            base = base.float().clamp_min(.001)
            text = self.text_encoder(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state
        r = self.residual(features, text, attention_mask, (ph, pw), (h, w))
        prediction = base * r.exp()
        return (prediction, base, r) if details else prediction


def residual_loss(base, residual, gt):
    if base.shape != gt.shape or residual.shape != gt.shape: raise ValueError('GT must not be resized')
    valid = torch.isfinite(gt) & (gt > .001) & (gt < 80.)
    losses = []
    for b, r, g, m in zip(base, residual, gt, valid):
        if m.any():
            if not torch.isfinite(b[m]).all() or not torch.isfinite(r[m]).all(): raise ValueError('Nonfinite prediction')
            losses.append((r[m].float() - (g[m].float().log()-b[m].float().log())).abs().mean())
    if not losses: raise ValueError('Entire microbatch has no valid GT')
    return torch.stack(losses).mean(), len(gt)-len(losses)
