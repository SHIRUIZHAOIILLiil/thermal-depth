"""Decoder-only adaptation primitives with a GT-free inference boundary."""
from dataclasses import dataclass
import copy

import torch

from tools.iris_tmax1_selection import select as select_reference

STEPS = (0, 500, 1000)
LR, BATCH, ACCUM, SEED = 1e-5, 4, 3, 43


@dataclass(frozen=True)
class InferenceInput:
    sample_id: str
    image_path: str
    caption: str
    manifest_index: int

    @classmethod
    def from_sample(cls, sample):
        return cls(sample.sample_id, sample.thermal_path, sample.caption, sample.manifest_line-1)


def predict_frozen(model, item, recipe):
    """No GT/path-to-GT, pseudo target, bounds or noisy-target argument accepted."""
    from tools.train_route_suite import load_input_tensor
    if type(item) is not InferenceInput or not item.caption.strip():
        raise ValueError('Expected a captioned, GT-free inference input')
    row = dict(id=item.sample_id, image_path=item.image_path, manifest_index=item.manifest_index)
    with torch.no_grad():
        image, _ = load_input_tensor(row, model.modality, recipe)
        prompt = model.encode_prompt(item.caption)
        raw, latent = model.predict_disparity(row, image, prompt,
            target_latent=None, return_latent=True, noise_offset=0)
    # no inference_mode tensors: the detached latent must support decoder backward.
    return raw.detach().clone(), latent.detach().clone()


def make_decoder(model):
    if model.gt_vae is None:
        raise ValueError('Parent must use its verified fp32 depth decoder')
    for module in (model.unet, model.lotus.vae, model.lotus.text_encoder, model.gt_vae):
        module.eval().requires_grad_(False)
    vae = copy.deepcopy(model.gt_vae).float().eval().requires_grad_(False)
    # Keep the mapping into the decoder and all encoder-side parameters frozen.
    vae.decoder.requires_grad_(True)
    params = {k:p for k,p in vae.named_parameters() if p.requires_grad}
    if not params or not all(k.startswith('decoder.') for k in params):
        raise ValueError('Unexpected trainable decoder scope')
    return vae, params


def decode_unit(vae, latent):
    with torch.autocast(device_type=latent.device.type, enabled=False):
        decoded = vae.decode(latent.float()/vae.config.scaling_factor, return_dict=False)[0]
    # Match deployment arithmetic order, including floating-point rounding.
    return (decoded.float()/2+.5).mean(dim=1, keepdim=True)


def select(curve, reference):
    if set(curve) != set(STEPS):
        raise ValueError('Expected decoder updates 0/500/1000')
    # Reuse the frozen policy without editing the historical tmax1 experiment.
    keys = {2000:0, 3000:500, 4000:1000}
    chosen, report = select_reference({k:curve[v] for k,v in keys.items()}, reference)
    report['selected'] = keys[chosen]
    report['candidates'] = {keys[k]:v for k,v in report['candidates'].items()}
    return keys[chosen], report
