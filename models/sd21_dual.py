"""Iris-G latent tasks plus a direct metric head. No teacher, no old-run edits."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from models.sd21_tadp import DepthHead
from models.da2_silog import masked_silog

LOSS_WEIGHTS = dict(latent_depth=1., reconstruction=1., image_l1=.1, direct_silog=1.)


def expand_input(unet):
    old = unet.conv_in
    if old.in_channels != 4: raise ValueError('Initialize from original four-channel SD2.1')
    new = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1,2,1,1)*.5)
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=8)


def task_labels(n, device):
    labels = torch.tensor([[1.,0.],[0.,1.]], device=device)
    labels = torch.cat((labels.sin(),labels.cos()),dim=-1)
    return labels.repeat_interleave(n,dim=0)


def decode_log_depth(vae, latent, bounds):
    # Exact l100 inverse, including NO clamp of decoded unit values.
    with torch.autocast(device_type=latent.device.type, enabled=False):
        decoded=vae.decode(latent.float()/vae.config.scaling_factor,return_dict=False)[0]
        y=decoded.float().mean(1)/2+.5
        lo,hi=bounds[:,0,None,None].float(),bounds[:,1,None,None].float()
        return (lo+y*(hi-lo+1e-5)).clamp(-9,9).exp()


def four_losses(latent_prediction, latent_target, decoded, direct, gt):
    n=gt.shape[0]
    if latent_prediction.shape != latent_target.shape or latent_prediction.shape[0] != 2*n:
        raise ValueError('Expected depth then reconstruction task batches')
    mask=torch.isfinite(gt)&(gt>.001)&(gt<80)
    if decoded.shape != gt.shape or direct.shape != gt.shape: raise ValueError('Depth shape mismatch')
    terms=dict(latent_depth=F.mse_loss(latent_prediction[:n].float(),latent_target[:n].float()),
               reconstruction=F.mse_loss(latent_prediction[n:].float(),latent_target[n:].float()),
               image_l1=(decoded[mask].float()-gt[mask].float()).abs().mean())
    terms['direct_silog'],empty=masked_silog(direct,gt)
    if not all(bool(torch.isfinite(x)) for x in terms.values()): raise ValueError('Nonfinite component loss')
    return sum(LOSS_WEIGHTS[k]*v for k,v in terms.items()),terms,empty


class DualDepth(nn.Module):
    def __init__(self,unet,vae,text_encoder,scheduler,empty_ids,head=None):
        super().__init__()
        self.unet=unet.requires_grad_(True)
        self.vae=vae.requires_grad_(False).eval()
        self.text_encoder=text_encoder.requires_grad_(False).eval()
        self.scheduler=scheduler
        self.head=head if head is not None else DepthHead()
        self.register_buffer('empty_ids',empty_ids.long().reshape(1,-1),persistent=False)
        if self.empty_ids.shape != (1,77):
            raise ValueError('Empty text must be padded to exactly 77 tokens')
        self._capture=None
        taps=[unet.up_blocks[0].resnets[1],unet.up_blocks[1].attentions[1],
              unet.up_blocks[2].attentions[1],unet.up_blocks[3].attentions[2]]
        self._handles=[m.register_forward_hook(self._hook(i)) for i,m in enumerate(taps)]

    def _hook(self,index):
        def hook(module,inputs,output):
            if self._capture is not None:
                self._capture[index]=output if torch.is_tensor(output) else output[0]
        return hook

    def train(self,mode=True):
        super().train(mode); self.vae.eval(); self.text_encoder.eval()
        return self

    def _predict(self,inputs,text,labels,n,size):
        self._capture=[None]*4
        try:
            latent=self.unet(inputs,torch.full((len(inputs),),999,device=inputs.device,dtype=torch.long),
                             encoder_hidden_states=text,class_labels=labels,return_dict=False)[0]
            if any(x is None for x in self._capture): raise RuntimeError('Missing feature tap')
            features=[x[:n] for x in self._capture[::-1]]
        finally:
            self._capture=None
        return latent,self.head(features,size)

    def _text(self,ids,mask,reconstruction):
        # All arms/tasks keep all77 positions. The tokenizer mask is metadata
        # only: preserve native causal CLIP and unmasked U-Net cross-attention.
        if ids.ndim!=2 or ids.shape[1]!=77 or mask.shape!=ids.shape:
            raise ValueError('Text and tokenizer mask must have shape B x 77')
        if reconstruction: ids=torch.cat((ids,self.empty_ids.expand(len(ids),-1)))
        return self.text_encoder(input_ids=ids).last_hidden_state

    def training_loss(self,image,ids,mask,target,bounds,gt,trace=False):
        n=len(image)
        with torch.no_grad(),torch.autocast(device_type=image.device.type,enabled=False):
            image_latent=self.vae.encode(torch.cat((image,image)).float()).latent_dist.sample()*self.vae.config.scaling_factor
            target_latent=self.vae.encode(torch.cat((target,image)).float()).latent_dist.sample()*self.vae.config.scaling_factor
            noise=torch.randn_like(target_latent)
            t=torch.full((2*n,),999,device=image.device,dtype=torch.long)
            noisy=self.scheduler.add_noise(target_latent,noise,t)
            text=self._text(ids,mask,True)
        inputs=torch.cat((image_latent,noisy),dim=1)
        latent,direct=self._predict(inputs,text,task_labels(n,image.device),n,image.shape[-2:])
        decoded=decode_log_depth(self.vae,latent[:n],bounds)
        loss,terms,empty=four_losses(latent,target_latent,decoded,direct,gt)
        stochastic=None
        if trace:
            h=hashlib.sha256()
            for value in (image_latent,target_latent,noise): h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
            stochastic=h.hexdigest()
        reconstruction_text=(hashlib.sha256(text[n:].detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() if trace else None)
        return loss,terms,dict(empty_gt_frames=empty,lidar_pixels=int((torch.isfinite(gt)&(gt>.001)&(gt<80)).sum()),stochastic=stochastic,reconstruction_text=reconstruction_text)

    def forward(self,image,token_ids,attention_mask):
        # Evaluation never consumes targets or bounds. Per-image seeds make
        # the one-step noise identical across arms and independent of batching.
        with torch.no_grad(),torch.autocast(device_type=image.device.type,enabled=False):
            z=self.vae.encode(image.float()).latent_dist.mode()*self.vae.config.scaling_factor
            noise=[]
            for x,zi in zip(image,z):
                seed=int.from_bytes(hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).digest()[:8],'little')%(2**63-1)
                g=torch.Generator(device=image.device).manual_seed(seed)
                noise.append(torch.randn(zi.shape,device=image.device,dtype=z.dtype,generator=g))
            text=self._text(token_ids,attention_mask,False)
        _,direct=self._predict(torch.cat((z,torch.stack(noise)),1),text,
                               task_labels(len(image),image.device)[:len(image)],len(image),image.shape[-2:])
        return direct


def load_dual(path):
    from diffusers import AutoencoderKL,UNet2DConditionModel,DDPMScheduler
    from transformers import CLIPTextModel,CLIPTokenizer
    unet=UNet2DConditionModel.from_pretrained(path,subfolder='unet',local_files_only=True,
        use_safetensors=True,low_cpu_mem_usage=False,device_map=None,
        class_embed_type='projection',projection_class_embeddings_input_dim=4)
    if unet.config.cross_attention_dim!=1024 or unet.config.in_channels!=4:
        raise ValueError('Requires original SD2.1-base')
    expand_input(unet); unet.enable_gradient_checkpointing()
    vae=AutoencoderKL.from_pretrained(path,subfolder='vae',local_files_only=True,use_safetensors=True)
    text=CLIPTextModel.from_pretrained(path,subfolder='text_encoder',local_files_only=True)
    tokenizer=CLIPTokenizer.from_pretrained(path,subfolder='tokenizer',local_files_only=True)
    empty=tokenizer([''],padding='max_length',max_length=77,return_tensors='pt')['input_ids']
    scheduler=DDPMScheduler.from_pretrained(path,subfolder='scheduler',local_files_only=True)
    scheduler.register_to_config(prediction_type='sample')
    return DualDepth(unet,vae,text,scheduler,empty)
