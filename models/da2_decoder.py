"""Frozen DINO/CLIP with trainable text-conditioned DPT inputs and decoder."""
import torch
from torch import nn
from torch.nn import functional as F


class DecoderFusion(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024, width=256):
        super().__init__()
        self.vnorm = nn.LayerNorm(vision_dim)
        self.tnorm = nn.LayerNorm(text_dim)
        self.visual = nn.Linear(vision_dim, width)
        self.text = nn.Linear(text_dim, width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.merge = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, vision_dim))
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)

    def forward(self, visual, text, mask):
        if mask.ndim != 2 or not mask.bool().any(dim=1).all():
            raise ValueError('Each caption needs at least one valid token')
        q = self.visual(self.vnorm(visual))
        kv = self.text(self.tnorm(text))
        attended, _ = self.attention(q, kv, kv, key_padding_mask=~mask.bool(), need_weights=False)
        return visual + self.merge(torch.cat((q, attended), dim=-1))


class DecoderDepth(nn.Module):
    def __init__(self, core, text_encoder, seed=43, vision_dim=768, text_dim=1024, width=256):
        super().__init__()
        self.core = core
        self.core.pretrained.requires_grad_(False).eval()
        self.core.depth_head.requires_grad_(True)
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.fusions = nn.ModuleList([DecoderFusion(vision_dim, text_dim, width) for _ in range(2)])

    def train(self, mode=True):
        super().train(mode)
        self.core.pretrained.eval()
        self.text_encoder.eval()
        return self

    def forward(self, image, token_ids=None, attention_mask=None):
        if token_ids is None or attention_mask is None:
            raise ValueError('Both arms require text and mask')
        h, w = image.shape[-2:]
        pad = (0, (-w)%14, 0, (-h)%14)
        x = F.pad(image, pad, mode='reflect') if any(pad) else image
        with torch.no_grad():
            features = list(self.core.pretrained.get_intermediate_layers(
                x, self.core.intermediate_layer_idx[self.core.encoder], return_class_token=True))
            text = self.text_encoder(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state
        for index, fusion in zip((2, 3), self.fusions):
            tokens, cls = features[index]
            features[index] = (fusion(tokens, text, attention_mask), cls)
        # Deliberately outside no_grad: both fusion and the full DPT head train.
        depth = self.core.depth_head(features, x.shape[-2]//14, x.shape[-1]//14)
        return (depth.squeeze(1)*self.core.max_depth)[..., :h, :w]
