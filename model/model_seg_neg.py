from audioop import bias
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from functools import reduce
from operator import add
from .extractor import Extractor
from .fuser import FeatureFuser
from utils.cam import cam_to_label, logits_to_label
from model.featurecluster import DFC_KL
from transformers import CLIPTextModel, CLIPTokenizer, Dinov2Model
from datasets.data_name_token import VOC12_NAME, COCO14_NAME

normalizer = T.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

class DINOMaskRefiner(nn.Module):
    def __init__(self, in_channels=768, beta=10, iter_num=3, alpha=0.5):
        super().__init__()
        self.beta = beta          # 控制对相似度的敏感度
        self.iter_num = iter_num  # 随机游走迭代次数
        self.alpha = alpha  # 融合比例:越小越信任 SD，越大越信任 DINO
   
    def get_affinity(self, features):

        B, C, H, W = features.shape

        flat_feats = features.view(B, C, -1) # [B, 128, N]
        flat_feats = F.normalize(flat_feats, p=2, dim=1)

        affinity = torch.bmm(flat_feats.permute(0, 2, 1), flat_feats) # [B, N, N]
        affinity = F.relu(affinity)
        affinity = torch.pow(affinity, self.beta)

        affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-6)

        return affinity

    def forward(self, pseudo_logits, dino_feat):
        """
        Args:
            coarse_mask: [B, K, H, W]  SD生成的伪标签(Logits或Prob) [1,21,500,333]
            dino_feat:   [B, C, h, w]  DINO提取的特征图 #[1,768,37,37]
        """
        h, w = dino_feat.shape[-2:]
        pseudo_logits = F.interpolate(pseudo_logits, size=(h, w), mode='bilinear', align_corners=False)

        probs = torch.softmax(pseudo_logits, dim=1) # [B, K, h, w]
        B, K, _, _ = probs.shape
        flat_probs = probs.view(B, K, -1) # [B, K, N]
        
        affinity = self.get_affinity(dino_feat)

        curr_probs = flat_probs
        for _ in range(self.iter_num):
            refined_flat = torch.bmm(curr_probs, affinity.transpose(1, 2))
            curr_probs = self.alpha * refined_flat + (1 - self.alpha) * curr_probs
        
        refined_prob = curr_probs.reshape(B, K, h, w)
        
        return torch.log(refined_prob + 1e-6)

class DiffusionBasedNetwork(nn.Module):
    def __init__(
            self, num_classes=21, backbone="stabilityai/stable-diffusion-2-1-base",
            block_indice = {
                'encoder': (),
                'unet': (5, 8, 11),
                'decoder': (),
            },
            attention_layers_to_use=[],
            no_use_self_ers=True,
            no_use_cross_enh=True,
            no_use_cluster=True,
            enhanced=1.6,
            ent=0.015,
            iter_num=10,
            cam_bg_thr=0,
            dataset_name='voc2012',
            feature_extractor="facebook/dinov2-with-registers-base"
        ): 

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.text_encoder = CLIPTextModel.from_pretrained(backbone, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(backbone, subfolder="tokenizer")

        print(f"Loading feature extractor model: {feature_extractor} ...")
        self.feature_extractor = Dinov2Model.from_pretrained(
            feature_extractor,
            attn_implementation="eager",
            output_hidden_states=True,
            output_attentions=True
        )

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.n_last_blocks = 4
        self.dino_embed_dim = self.feature_extractor.config.hidden_size

        self.num_classes_cls = num_classes - 1
        self.num_classes = num_classes

        self.no_use_self_ers = no_use_self_ers
        self.no_use_cross_enh = no_use_cross_enh
        self.no_use_cluster = no_use_cluster
        self.att_mean = True
        self.enhanced = enhanced
        self.cam_bg_thr = cam_bg_thr
        self.iter_num = iter_num
        self.ent = ent
        self.dataset_name = dataset_name
        
        self.bg_context = None
        self.index = {32: [22, 5, 21, 13, 8, 27, 28, 6, 25],
                      16: [76, 1, 43, 17, 81, 6, 44, 27, 2, 8, 22, 20, 60, 78, 12, 83, 94, 47,
                           88, 96, 3, 33, 46, 52, 77, 93, 51, 58, 0, 13, 14, 19, 34, 41, 59, 87,
                           16, 24, 28, 30, 32]} if self.att_mean else None
        self.class_name = []
        self.all_tokens = {}

        self.prepare_data_name()

        self.fuser = FeatureFuser(
            backbone=backbone,
            block_indices=block_indice,
            attention_layers_to_use=attention_layers_to_use,
        )

        self.dino_refiner = DINOMaskRefiner(in_channels=self.dino_embed_dim, beta=10, iter_num=4, alpha=0.3)

        fused_feature, _, _ = self.fuser(torch.randn(1, 3, 512, 512), '')
        seg_in_channels = fused_feature.shape[1]  # 返回单个特征张量
    
        self.pseudo_mask_generator = nn.Sequential(
            nn.Conv2d(seg_in_channels, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )

        self.dino_projector = nn.Sequential(
            nn.Conv2d(self.dino_embed_dim * self.n_last_blocks, self.dino_embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, self.dino_embed_dim),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(self.dino_embed_dim, 256, 3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.pseudo_mask_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def prepare_data_name(self):
        if self.dataset_name == 'coco2014':
            self.all_tokens = COCO14_NAME 
            self.bg_context = ("ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, "
                             "railroad, helmet, cloud, house, mountain, ocean, road, rock, street, valley, bridge")

        elif self.dataset_name == 'voc2012':
            self.all_tokens = VOC12_NAME
            # self.bg_context = ("ground, land, grass, tree, building, wall, sky, lake, water"
            #                  ", river, sea, railway, railroad, keyboard, helmet, cloud, house"
            #                  ", mountain, ocean, road, rock, street, valley, bridge, sign")
            self.bg_context = "ground, land, grass, tree, building, wall, sky, lake, water"
                             
        self.class_name = list(self.all_tokens.keys())

    def process_cross_att(self, cross_attention_maps):

        weight_layer = {8: 0.0, 16: 0.7, 32: 0.3, 64: 0}
        cross_attention = []
        for key, values in cross_attention_maps.items():
            if len(values) == 0 or key in [8, 64]: continue
            if self.index is None:
                values = values.mean(1) # 使用所有注意力头的平均
            else:
                values = values[:, self.index[key]].mean(1)
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True) # 对每个注意力图进行归一化，确保所有像素权重和为 1
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode='bilinear', align_corners=False) # 所有注意力图上采样为64*64
            cross_attention.append(weight_layer[key] * normed_attn)  
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]  # 堆叠并相加 然后取第0个batch
        if self.no_use_cluster: #可选的特征聚类
            dfc = DFC_KL(32, 20, 64)
            clusters, n = dfc(cross_attention)
            one_hot = F.one_hot(clusters, n)
            self_att = one_hot[:, clusters]
            cross_attention = torch.matmul(self_att.type(cross_attention.dtype),cross_attention.flatten(-2, -1).permute(1, 0))
            cross_attention /= self_att.sum(-1, keepdim=True)
        else:
            cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)
        cross_attention = torch.stack([cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1)
        
        return cross_attention[None]

    def get_att_map(self, cross_attention_maps, self_attention_maps):

        if not self.no_use_self_ers:
            return super().get_att_map(cross_attention_maps, self_attention_maps)
        else:
            # cross attention 特征归一化 & 上采样融合
            cross_att = self.process_cross_att(cross_attention_maps).float()
            cross_attn = cross_att - cross_att.amin(dim=-2, keepdim=True)  # cross_att: 4096, 20
            cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化

            trans_mat = self_attention_maps[64][:, [1, 2]].mean(1).flatten(-2, -1).permute(0, 2, 1).float()
            trans_mat /= torch.amax(trans_mat, dim=-2, keepdim=True)

            trans_mat += torch.where(trans_mat == 0, 0, self.ent * (torch.log10(torch.e * trans_mat)))
            trans_mat = torch.clamp(trans_mat, min=0)

            trans_mat_p = trans_mat.clone()
            trans_mat_p /= trans_mat_p.sum(dim=-1, keepdim=True)

            for _ in range(self.iter_num):
                cross_attn = torch.bmm(trans_mat_p, cross_attn)
                # cross_attn = torch.where(cross_attn < cross_attn.amax(dim=-2, keepdim=True) * 0.1, 0, cross_attn)
                cross_attn -= cross_attn.amin(dim=-2, keepdim=True)
                cross_attn /= cross_attn.sum(dim=-2, keepdim=True)

            cross_att = cross_attn
        att_map = cross_att.unflatten(dim=-2, sizes=(64, 64)).permute(0, 3, 1, 2)
        att_map = F.interpolate(att_map, size=512, mode='bilinear', align_corners=False)
        att_map = att_map[0]
        att_map -= att_map.amin(dim=(-2, -1), keepdim=True)
        att_map /= att_map.amax(dim=(-2, -1), keepdim=True)

        return att_map

    def get_text_embedding(self, text: str) -> torch.Tensor:

        text_input = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors="pt")

        with torch.set_grad_enabled(False):
            embedding = self.text_encoder(text_input.input_ids.to(self.device), output_hidden_states=True)[0]

        return embedding

    def prepare_text_embeddings(self, active_classes):

        self.token_start_ids = []
        self.token_sel_ids = []

        img_label = [self.class_name[i] for i in active_classes]

        text = "a photograph of "
        last_cls = None
        for idx, cls in enumerate(img_label):
            text += cls + " and "
            self.token_start_ids.append( # token_start_ids[0] = 4 起始位置
                self.token_start_ids[idx - 1] + 1 + self.all_tokens[last_cls][0] if idx > 0 else 4)
            self.token_sel_ids.append([self.token_start_ids[-1] + sel_id for sel_id in self.all_tokens[cls][1]])
            last_cls = cls

        text = text[:-5] + " and other object and background, " + self.bg_context
        text_embedding = self.get_text_embedding(text)
        
        # 增强特定token的权重
        meaning_index = reduce(add, self.token_sel_ids)
        text_embedding[:, meaning_index] *= self.enhanced if self.no_use_cross_enh else 1
            
        return text_embedding

    def extract_feature(self, x):
        B, _, H, W = x.shape
        
        outputs = self.feature_extractor(x)

        patch_size = 14
        h_grid = H // patch_size
        w_grid = W // patch_size
        num_patches = h_grid * w_grid

        # 提取最后一层 Patch Tokens
        hidden_states = outputs.hidden_states

        last_n_states = hidden_states[-self.n_last_blocks:] 

        cat_feat = []
        for state in last_n_states:
            feat = state[:, -num_patches:, :]
            feat = feat.permute(0, 2, 1).reshape(B, self.dino_embed_dim, h_grid, w_grid)
            cat_feat.append(feat)

        fused_feat = torch.cat(cat_feat, dim=1)

        dino_feat = self.dino_projector(fused_feat)

        return dino_feat

    def forward(self, x, cls_label, mask=None, img_name=None):
        # 准备文本嵌入
        active_classes = torch.where(cls_label[0] == 1)[0]
        text_embedding = self.prepare_text_embeddings(active_classes) # [1,77,1024]

        # 提取融合特征用于生成 pseudo_mask
        fused_feature, cross_attention_maps, self_attention_maps = self.fuser(x, text_embedding) 

        # 获取注意力图
        # att_map = self.get_att_map(cross_attention_maps, self_attention_maps) 
        # final_attention_map = torch.zeros(self.num_classes_cls, 512, 512, device=x.device)
        # final_attention_map[active_classes] += att_map 
        # final_attention_map = F.interpolate(final_attention_map[None], size=mask.shape[-2:], mode="bilinear", align_corners=False)[0] 
        # valid_cam, attn_label = cam_to_label(final_attention_map[None].clone(), cls_label=cls_label, bkg_thre=self.cam_bg_thr) 

        # 生成的伪预测
        pseudo_logits_low = self.pseudo_mask_generator(fused_feature)  # [1,21,64,64]
        # pseudo_logits = F.interpolate(pseudo_logits_low, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        x_dino = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
        dino_feat = self.extract_feature(normalizer(x_dino)) 

        # 用 DINO 优化伪预测得到优化后的伪标签
        refined_pseudo_logits = self.dino_refiner(pseudo_logits_low.detach(), dino_feat.detach()) # [1,21,64,64]
        refined_pseudo_logits = F.interpolate(refined_pseudo_logits, size=mask.shape[-2:], mode='bilinear', align_corners=False) 
        refined_label = logits_to_label(refined_pseudo_logits, cls_label=cls_label) #[1,500,333]
        # refined_label = logits_to_label(pseudo_logits, cls_label=cls_label) #[1,500,333]

        # 纯视觉分支前向
        pred = self.seg_head(dino_feat) 
        pred = F.interpolate(pred, size=mask.shape[-2:], mode='bilinear', align_corners=False) 

        # return pseudo_logits, attn_label, pred, refined_label
        return refined_label
    
    def get_param_groups(self):
        param_group = [[], [], []] # feature_fuser, pseudo_mask_generator, seg_head

        # 特征融合网络
        for param in self.fuser.parameters():
            param_group[0].append(param)
        # 伪掩码生成器参数
        for param in self.pseudo_mask_generator.parameters():
            param_group[1].append(param)
        # 分割头参数
        for param in self.seg_head.parameters():
            param_group[2].append(param)

        for param in self.dino_projector.parameters():
            param_group[2].append(param)
            
        for param in self.dino_refiner.parameters():
            param_group[2].append(param)
        
        return param_group