"""Official-form SiLog and fully trainable DA2, with optional caption fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from models.da2_decoder import DecoderFusion


class SiLogDepth(nn.Module):
    def __init__(self, core, text_encoder=None, seed=43, vision_dim=768, text_dim=1024, width=256):
        super().__init__()
        self.core = core.requires_grad_(True)
        self.text_encoder = text_encoder
        self.fusions = nn.ModuleList()
        if text_encoder is not None:
            text_encoder.requires_grad_(False).eval()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.fusions.extend([DecoderFusion(vision_dim, text_dim, width) for _ in range(2)])

    def train(self, mode=True):
        super().train(mode)
        if self.text_encoder is not None: self.text_encoder.eval()
        return self

    def forward(self, image, token_ids=None, attention_mask=None):
        h,w = image.shape[-2:]
        pad=(0,(-w)%14,0,(-h)%14)
        x=F.pad(image,pad,mode='reflect') if any(pad) else image
        if self.text_encoder is None:
            return self.core(x)[...,:h,:w]
        if token_ids is None or attention_mask is None: raise ValueError('Caption requires tokens and mask')
        with torch.no_grad():
            text=self.text_encoder(input_ids=token_ids,attention_mask=attention_mask).last_hidden_state
        # Encoder must remain in the gradient graph for both arms.
        features=list(self.core.pretrained.get_intermediate_layers(
            x,self.core.intermediate_layer_idx[self.core.encoder],return_class_token=True))
        for i,fusion in zip((2,3),self.fusions):
            tokens,cls=features[i]
            features[i]=(fusion(tokens,text,attention_mask),cls)
        depth=self.core.depth_head(features,x.shape[-2]//14,x.shape[-1]//14)
        return (depth.squeeze(1)*self.core.max_depth)[...,:h,:w]


def masked_silog(prediction, gt):
    """Official lambda=.5 formula, pooled valid pixels per microbatch, FP32.

    Retain thermal experiment's strict GT mask (0.001,80). No GT interpolation,
    prediction clipping, alignment, per-frame averaging, or extra loss terms.
    Exact-zero error has a zero subgradient to avoid sqrt(0)'s undefined gradient.
    """
    if prediction.shape != gt.shape: raise ValueError('Prediction/GT shapes differ')
    valid=torch.isfinite(gt) & (gt>.001) & (gt<80.)
    if not valid.any(): raise ValueError('Entire microbatch has no valid LiDAR')
    p=prediction[valid].float(); g=gt[valid].float()
    if not torch.isfinite(p).all() or not (p>0).all(): raise ValueError('SiLog requires finite positive predictions')
    e=g.log()-p.log()
    value=e.square().mean()-.5*e.mean().square()
    loss=e.sum()*0 if bool((e==0).all()) else value.sqrt()
    empty=int((~valid.flatten(1).any(1)).sum())
    return loss,empty
