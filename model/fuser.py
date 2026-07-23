import torch
import torch.nn as nn
import torch.nn.functional as F
from model.extractor import Extractor

class FeatureFuser(nn.Module):
    def __init__(
            self, 
            backbone="stabilityai/stable-diffusion-2-1-base",
            block_indices = {
                'encoder': (),
                'unet': (5, 8, 11),
                'decoder': (),
            },
            attention_layers_to_use=[],
        ):
        
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.extractor = Extractor(
            model_name=backbone,
            block_indices=block_indices,
            attention_layers_to_use=attention_layers_to_use,
        )

        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        dummy_text = ''

        with torch.no_grad():
            features, _, _ = self.extractor(dummy_input, dummy_text)

        self.projections = nn.ModuleList()
        self.dims = [256, 128, 64]
        
        for i, feat in enumerate(features):
            in_channels = feat.shape[1]
            out_channels = self.dims[i]
           
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32 if out_channels >= 32 else 8, out_channels), # 动态 Group 数
                    nn.ReLU(inplace=True)
                )
            )

        self.target_dim = sum(self.dims) 

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.target_dim+3, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.to(self.device)

    def forward(self, input_image, text_embedding):
        
        features, cross_attention_maps, self_attention_maps = self.extractor(input_image, text_embedding)

        projected_features = []

        for feat, projector in zip(features, self.projections):
            projected_features.append(projector(feat))
        
        target_h = max([f.shape[2] for f in projected_features])
        target_w = max([f.shape[3] for f in projected_features]) # 64*64

        # target_h = 128
        # target_w = 128

        resized_features = []
        for feat in projected_features:
            if feat.shape[2:] != (target_h, target_w):
                feat_resized = F.interpolate(
                    feat, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                feat_resized = feat
            resized_features.append(feat_resized)

        input_image_resized = F.interpolate(input_image,size=(target_h, target_w), mode='bilinear',align_corners=False).to(self.device)
        resized_features.append(input_image_resized)

        fused_with_rgb = torch.cat(resized_features, dim=1)

        fused_feature = self.final_conv(fused_with_rgb)

        return fused_feature, cross_attention_maps, self_attention_maps
