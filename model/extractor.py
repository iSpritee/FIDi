import sys
import os

from pylab import False_
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from utils.attn import AttnCLusterProcessor
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import CLIPTextModel, CLIPTokenizer

class Extractor(nn.Module):
    def __init__(
            self, model_name = "Manojb/stable-diffusion-2-1-base",
            block_indices = {
                'encoder': (2, 5),
                'unet': (2, 5, 8, 11),
                'decoder': (5, 7),
            },
            timesteps=(0, 200),
            attention_layers_to_use=[]
        ):
                 
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] loading stable diffusion extractor features...")
        
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 设置特征提取参数
        self.block_indices = block_indices
        self.timesteps=timesteps
        self.attention_layers_to_use = attention_layers_to_use

        self.attention_maps = {}

        def create_nested_hook_for_attention_modules(n):
            def hook(module, input, output):
                bs_head, h, w = output[1].shape
                self.attention_maps[n] = output[1].reshape(bs_head // module.heads, module.heads, h, w)
            return hook

        self.handles = []
        self.processor = AttnCLusterProcessor()
        for module in self.attention_layers_to_use:
            self.handles.append(
                eval("self.unet." + module).register_forward_hook(
                    create_nested_hook_for_attention_modules(module)
                )
            )
            eval("self.unet." + module).processor = self.processor
       
        rng = torch.Generator().manual_seed(42)
        self.register_buffer(
            "shared_noise",
            torch.randn(1, 4, 64, 64, generator=rng),
        )

        self.to(self.device)
        self.eval()

    def get_text_embedding(self, text: str) -> torch.Tensor:

        text_input = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt")

        with torch.set_grad_enabled(False):
            embedding = self.text_encoder(text_input.input_ids.cuda(), output_hidden_states=True)[0]

        return embedding

    def get_attention_map(self):

        raw_attention_maps = self.attention_maps  

        cross_attention_maps = {8: [], 16: [], 32: [], 64: []}
        self_attention_maps = {8: [], 16: [], 32: [], 64: []}

        for layer in raw_attention_maps: 
            bs, head, img_embed_len, text_embed_len = raw_attention_maps[layer].shape
            hw = int(math.sqrt(img_embed_len)) 
            reshaped_attn = raw_attention_maps[layer].reshape(bs, head, hw, hw, text_embed_len).softmax(-1) 
            if layer.endswith("attn2"):  # cross attentions
                cross_attention_maps[hw].append(reshaped_attn)
            elif layer.endswith("attn1"):  # self attentions
                self_attention_maps[hw].append(reshaped_attn)

        for key, values in cross_attention_maps.items(): 
            if len(values) == 0: continue
            attn = torch.cat(values, dim=1)
            attn = attn.permute(0, 1, 4, 2, 3) # (batch_size, total_heads, text_len, height, width)
            cross_attention_maps[key] = attn

        for key, values in self_attention_maps.items(): 
            if len(values) == 0: continue
            attn = torch.cat(values, dim=1)
            attn = attn.permute(0, 1, 4, 2, 3)
            self_attention_maps[key] = attn

        return cross_attention_maps, self_attention_maps

    def encoder_forward(self, x):
        """VAE编码器前向传播"""
        encoder = self.vae.encoder
        features = []

        x = x * 2 - 1
        h = encoder.conv_in(x)

        block_idx = 0
        # downsampling 
        for i, down_block in enumerate(encoder.down_blocks):
            for j, resnet in enumerate(down_block.resnets):
                # 在处理前提取特征
                if block_idx in self.block_indices['encoder']:
                   features.append(h.contiguous())

                h = resnet(h, None)   
                block_idx += 1

            if hasattr(down_block, 'downsamplers') and down_block.downsamplers is not None:
                for downsampler in down_block.downsamplers:
                    h = downsampler(h)
        
        # 中间块处理
        if encoder.mid_block is not None:
            h = encoder.mid_block(h)
        
        h = encoder.conv_norm_out(h)
        h = encoder.conv_act(h)  # Swish
        h = encoder.conv_out(h)  # 最终处理步骤
      
        
        moments = self.vae.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.mean * self.vae.config.scaling_factor

        return latents, features
    
    def unet_forward(self, x, timesteps, context):
        unet = self.unet
        features = []
        hs = []

        # 时间步嵌入
        timesteps = timesteps.to(dtype=torch.float32)
        t_emb = unet.time_proj(timesteps)
        t_emb = self.unet.time_embedding(t_emb)
        
        h = self.unet.conv_in(x)
        hs.append(h)  # 存储第1个特征
        
        # Input Blocks 1-10: 遍历所有下采样块
        for i, down_block in enumerate(unet.down_blocks):
            # 处理每个 ResNet 块
            for j, resnet in enumerate(down_block.resnets):
                h = resnet(h, t_emb)
                # 处理对应的 Attention 块
                if hasattr(down_block, 'attentions') and j < len(down_block.attentions):
                    h = down_block.attentions[j](
                        h,
                        encoder_hidden_states=context,
                        cross_attention_kwargs=None,
                    ).sample
                hs.append(h) 
            
            if hasattr(down_block, 'downsamplers') and down_block.downsamplers:
                h = down_block.downsamplers[0](h)
                hs.append(h)  

        if unet.mid_block is not None:
            h = unet.mid_block(h, t_emb, encoder_hidden_states=context)
        
        block_idx = 0
        for i, up_block in enumerate(unet.up_blocks):
            for resnet_idx, resnet in enumerate(up_block.resnets):
                if hs:
                    h = torch.cat([h, hs.pop()], dim=1)
                if block_idx in self.block_indices['unet']:
                    features.append(h.contiguous())
                h = resnet(h, t_emb)
                if hasattr(up_block, 'attentions') and up_block.attentions and resnet_idx < len(up_block.attentions):
                    h = up_block.attentions[resnet_idx](
                        h,
                        encoder_hidden_states=context,
                        cross_attention_kwargs=None,
                    ).sample
                block_idx += 1
            if hasattr(up_block, 'upsamplers') and up_block.upsamplers:
                h = up_block.upsamplers[0](h)
        
        h = unet.conv_norm_out(h)
        h = unet.conv_act(h)
        h = unet.conv_out(h)
        
        return h, features
    
    def decoder_forward(self, latents):
        decoder = self.vae.decoder
        features = []
        
        z = 1.0 / self.vae.config.scaling_factor * latents
        h = self.vae.post_quant_conv(z)
        
        h = decoder.conv_in(h)
        h = decoder.mid_block(h)

        block_idx = 0

        for i, up_block in enumerate(decoder.up_blocks):
            for j, resnet in enumerate(up_block.resnets):
                if block_idx in self.block_indices['decoder']:
                    features.append(h.contiguous())
                
                h = resnet(h, None)      
                if hasattr(up_block, "attentions") and j < len(up_block.attentions): 
                    h = up_block.attentions[j](h) 
                block_idx += 1

            if hasattr(up_block, "upsamplers") and up_block.upsamplers is not None:
                h = up_block.upsamplers[0](h)
        
        h = decoder.conv_norm_out(h)
        h = decoder.conv_act(h)
        image = decoder.conv_out(h)
        
        return image, features
    
    @torch.no_grad()
    def forward(self, x, text_embedding):

        batch_size = x.shape[0]

        self.attention_maps = {}

        input_image = x.to(self.device)  
        if not (input_image.shape[-2] == 512 and input_image.shape[-1] == 512):
            input_image = F.interpolate(input_image, (512, 512), mode="bilinear", align_corners=False)

        latent_image, encoder_features = self.encoder_forward(input_image)

        if not isinstance(text_embedding, torch.Tensor):
            text_embedding = self.get_text_embedding(text_embedding)
        
        # 2. UNet forward for multiple steps
        unet_features = []
        noise = self.shared_noise.expand_as(latent_image).to(latent_image.device)

        for t in self.timesteps:

            if t == 0:
                t = torch.full((batch_size,), t, dtype=torch.long, device=latent_image.device)
                _, step_unet_features = self.unet_forward(latent_image, t, text_embedding)
                unet_features.extend(step_unet_features)
            else:
                t = torch.full((batch_size,), t, dtype=torch.long, device=latent_image.device)
                self.attention_maps = {}
                latents_noisy = self.scheduler.add_noise(latent_image, noise, t)  
                _, step_unet_features = self.unet_forward(latents_noisy, t, text_embedding)

                cross_attention_maps, self_attention_maps = self.get_attention_map()

        self.scheduler.set_timesteps(1000)

        all_features = unet_features
        # [1920,16,16]、[960,32,32]、[640,64,64] 

        return all_features, cross_attention_maps, self_attention_maps
