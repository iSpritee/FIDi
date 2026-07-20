import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False, img_box=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def logits_to_label(logits, cls_label, ignore_index=255):
    """
    将包含背景通道的 Logits 转换为 Label，利用 cls_label 过滤噪声。
    Args:
        logits: [B, 21, H, W], 包含背景(idx 0)
        cls_label: [B, 20], 0/1 标签，不含背景
    """
    B, C, H, W = logits.shape
    
    bg_label = torch.ones((B, 1), device=cls_label.device)
    cls_label_with_bg = torch.cat([bg_label, cls_label], dim=1) # [B, 21]
    
    # 扩展到 [B, 21, H, W]
    cls_mask = cls_label_with_bg.unsqueeze(-1).unsqueeze(-1) # [B, 21, 1, 1]
    cls_mask = cls_mask.expand_as(logits)
    
    # 2. 过滤 Logits
    # 对于图像中不存在的类别，将其 Logits 设为极小值 (-inf)，确保 Argmax 不会选中它
    masked_logits = logits.clone()
    masked_logits = masked_logits.masked_fill(cls_mask == 0, -1e9)
    
    # 3. 生成伪标签 (直接 Argmax)
    pseudo_label = torch.argmax(masked_logits, dim=1) # [B, H, W]
    
    return pseudo_label


def cam_to_label(cam, cls_label, bkg_thre=0.3, cls_thre=0.4):
    """
    Args:
        cam: size-->[bs, n_cls, h, w] without the background
        cls_label: size-->[bs, n_cls] identify the class id of this picture without background
        bkg_thre: float, identify the min thresh of a class cam [0-1]
        cls_thre: float, identify the classification thresh [0-1]
    Returns:
        _pseudo_label: size-->[bs, h, w] the class label map with background
    """
    b, c, h, w = cam.shape

    cls_label_rep = (cls_label > cls_thre).unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w]) #[1,20,500,333]
    reshape_cam = cam.reshape(b, c, -1) # [1,20,500*333=166500]
    reshape_cam -= reshape_cam.amin(dim=-1, keepdim=True)  # 归一化
    reshape_cam /= reshape_cam.amax(dim=-1, keepdim=True) + 1e-6
    if bkg_thre > 0:
        reshape_cam[reshape_cam < bkg_thre] = 0
        reshape_cam = reshape_cam.reshape(b, c, h, w)

        valid_cam = cls_label_rep * reshape_cam
        cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
        _pseudo_label += 1
        _pseudo_label[cam_value == 0] = 0
    else:
        reshape_cam = reshape_cam.reshape(b, c, h, w) # [1,20,500,333]
        valid_cam = cls_label_rep * reshape_cam  # [1,20,500,333]  其他维度都为 0
        valid_cam = torch.cat([torch.pow(1 - valid_cam.amax(dim=1, keepdim=True), 2), valid_cam], dim=1) #[1,21,500,333]
        _pseudo_label = valid_cam.argmax(dim=1)
              
    return valid_cam, _pseudo_label


def get_kernel():
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight


class PAR(nn.Module):

    def __init__(self, dilations, num_iter, ):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d] * 4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b * c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)

        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)

        for d in self.dilations:
            pos_xy.append(ker * d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1, 1, _imgs.shape[self.dim], 1, 1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1) ** 2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1) ** 2
        # pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
