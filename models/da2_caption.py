"""DA V2 metric-depth wrapper. No-caption creates NO text/adapter parameters."""
import torch
from torch import nn
import torch.nn.functional as F


class CaptionAdapter(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024, width=256, heads=4):
        super().__init__()
        self.visual_norm = nn.LayerNorm(vision_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.query = nn.Linear(vision_dim, width)
        self.key_value = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.output = nn.Linear(width, vision_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, visual, text, mask):
        if mask.ndim != 2 or not mask.bool().any(dim=1).all():
            raise ValueError('Every caption must contain at least one unmasked token')
        q = self.query(self.visual_norm(visual))
        kv = self.key_value(self.text_norm(text))
        delta, _ = self.attention(q, kv, kv, key_padding_mask=~mask.bool(), need_weights=False)
        return visual + self.alpha * self.output(delta)


class CaptionDepth(nn.Module):
    def __init__(self, core, text_encoder=None, adapter_seed=43, vision_dim=768, text_dim=1024):
        super().__init__()
        self.core = core
        self.text_encoder = text_encoder
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            # Constructing adapters must not change backbone/data RNG streams.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(adapter_seed)
                self.adapters = nn.ModuleList([
                    CaptionAdapter(vision_dim, text_dim), CaptionAdapter(vision_dim, text_dim)])
        else:
            self.adapters = nn.ModuleList()

    def train(self, mode=True):
        super().train(mode)
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def forward(self, image, token_ids=None, attention_mask=None):
        h, w = image.shape[-2:]
        pad = (0, (-w) % 14, 0, (-h) % 14)
        x = F.pad(image, pad, mode='reflect') if any(pad) else image
        if self.text_encoder is None:
            depth = self.core(x)
        else:
            if token_ids is None or attention_mask is None:
                raise ValueError('Caption arm requires token IDs and padding mask')
            with torch.no_grad():
                text = self.text_encoder(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state
            features = list(self.core.pretrained.get_intermediate_layers(
                x, self.core.intermediate_layer_idx[self.core.encoder], return_class_token=True))
            for index, adapter in zip((2, 3), self.adapters):
                tokens, cls_token = features[index]
                features[index] = (adapter(tokens, text, attention_mask), cls_token)
            depth = self.core.depth_head(features, x.shape[-2] // 14, x.shape[-1] // 14)
            depth = depth.squeeze(1) * self.core.max_depth
        return depth[..., :h, :w]


def masked_log_l1(prediction, gt):
    if prediction.shape != gt.shape:
        raise ValueError('prediction/GT shapes differ; sparse GT must not be resized')
    valid = torch.isfinite(gt) & (gt > .001) & (gt < 80.)
    per_frame = []
    for p, g, m in zip(prediction, gt, valid):
        if m.any():
            if not torch.isfinite(p[m]).all():
                raise ValueError('Nonfinite prediction at valid GT')
            per_frame.append((p[m].float().clamp_min(.001).log() - g[m].float().log()).abs().mean())
    if not per_frame:
        raise ValueError('Entire micro-batch has no valid LiDAR')
    return torch.stack(per_frame).mean(), len(gt) - len(per_frame)
