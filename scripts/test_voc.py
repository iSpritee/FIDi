import argparse
import datetime
import logging
import os
import random
import sys
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保路径包含当前目录
sys.path.append(".")

# 导入自定义模块 (请确保这些模块存在)
from datasets import voc as voc
from model.model_seg_neg import DiffusionBasedNetwork
from utils import evaluate
from utils.dcrf import crf_inference
from utils.pyutils import AverageMeter, format_tabs

warnings.filterwarnings('ignore')
torch.hub.set_dir("./pretrained")

parser = argparse.ArgumentParser()
parser.add_argument("--diffusion_model", default='stabilityai/stable-diffusion-2-1-base', type=str)
parser.add_argument("--data_folder", default='/media/store1/xln/DiffusionCAM/VOCdevkit/VOC2012', type=str)
parser.add_argument("--list_folder", default='datasets/voc', type=str)
parser.add_argument("--num_classes", default=21, type=int)
parser.add_argument("--ignore_index", default=255, type=int)
parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str)
parser.add_argument("--train_set", default="train_aug", type=str)
parser.add_argument("--val_set", default="val", type=str)
parser.add_argument("--spg", default=1, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument('--backend', default='nccl')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--use_tta", action="store_true", default=True)
parser.add_argument("--use_crf", action="store_true", default=False) # 默认关闭CRF以加快速度
parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=[
            'down_blocks[1].attentions[1].transformer_blocks[0].attn2',
            'down_blocks[2].attentions[0].transformer_blocks[0].attn2',
            'down_blocks[2].attentions[1].transformer_blocks[0].attn2',
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",
            'up_blocks[1].attentions[2].transformer_blocks[0].attn1',
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",
            'mid_block.attentions[0].transformer_blocks[0].attn1',
        ],
    )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_precision_recall_scores(gts, preds, num_classes, ignore_index=255):
    """
    通过计算混淆矩阵来获取每类的 Precision 和 Recall
    """
    hist = np.zeros((num_classes, num_classes))
    
    # print("Calculating Precision & Recall...") # 减少刷屏
    for gt, pred in zip(gts, preds):
        gt = gt.flatten()
        pred = pred.flatten()
        mask = (gt != ignore_index)
        
        gt_valid = gt[mask].astype(int)
        pred_valid = pred[mask].astype(int)
        
        count = np.bincount(
            num_classes * gt_valid + pred_valid,
            minlength=num_classes ** 2
        )
        hist += count.reshape(num_classes, num_classes)

    tp = np.diag(hist)
    sum_pred = hist.sum(axis=0)
    sum_gt = hist.sum(axis=1)

    precision = np.divide(tp, sum_pred, out=np.zeros_like(tp), where=sum_pred!=0)
    recall = np.divide(tp, sum_gt, out=np.zeros_like(tp), where=sum_gt!=0)

    return precision, recall

def decode_segmap(label_mask, num_classes=21, ignore_index=255):
    palette = np.array([
        [0, 0, 0],        # background
        [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128]
    ])

    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(num_classes):
        rgb[label_mask == cls] = palette[cls]

    rgb[label_mask == ignore_index] = [255, 255, 255]

    return Image.fromarray(rgb)

def print_pr_table(precision, recall, class_names, title="P/R Table"):
    """辅助函数：打印漂亮的表格"""
    print(f"\n=== {title} ===")
    print(f"{'Class Name':<20} | {'Precision':<15} | {'Recall':<15}")
    print("-" * 60)
    
    mPrec_list = []
    mRec_list = []
    
    # 假设 class_names 包含了 num_classes 个名字
    # 如果 precision 长度 (21) 大于 class_names 长度，需要防止越界
    num_classes = len(precision)

    for i in range(num_classes):
        c_name = class_names[i] if i < len(class_names) else f"Class {i}"
        
        p_val = precision[i] * 100
        r_val = recall[i] * 100
        
        print(f"{c_name:<20} | {p_val:>10.2f} %    | {r_val:>10.2f} %")
        
        mPrec_list.append(precision[i])
        mRec_list.append(recall[i])

    print("-" * 60)
    print(f"{'Mean (All)':<20} | {np.mean(mPrec_list)*100:>10.2f} %    | {np.mean(mRec_list)*100:>10.2f} %")
    print("=" * 60 + "\n")

def validate(args=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    args.local_rank = local_rank

    if args.local_rank == 0:
        logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    # 1. 使用 train_aug 数据集
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split='train', 
        stage='val',
        rescale=None,
        hor_flip=False,
        crop_method='interpolate'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    device = torch.device(args.local_rank)
    
    model = DiffusionBasedNetwork(
        num_classes=args.num_classes,
        backbone=args.diffusion_model,
        attention_layers_to_use=args.attention_layers_to_use,
        dataset_name='voc2012'
    )

    model = model.to(device)
    checkpoint_path = 'work_dir_voc/5811/checkpoints/model_iter_46000.pth'
    
    if args.local_rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
        print(f"Configurations -> TTA: {args.use_tta}, CRF: {args.use_crf}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)    

    preds, gts, _pseudo_labels = [], [], []

    model.eval()
    # 创建CAM保存目录

    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    pred_dir = os.path.join(base_dir, "Train_Pred", args.val_set)  
    os.makedirs(pred_dir, exist_ok=True)
    
    scales = [1.0, 1.5, 1.25] if args.use_tta else [1.0]

    with torch.no_grad():
        # 只在主进程显示验证进度条
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >=", 
                           desc='Validating', disable=args.local_rank != 0):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            H, W = labels.shape[1], labels.shape[2]

            preds_logit_list = []
            
            for scale in scales:
                if scale != 1.0:
                    scaled_inputs = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    scaled_inputs = inputs
                
                pseudo_logits = model(scaled_inputs, cls_label, labels)
                # _, _, pred, _ = model(inputs, cls_label, labels)
                
                segs = F.interpolate(pseudo_logits, size=(H, W), mode='bilinear', align_corners=False)
                preds_logit_list.append(segs)

                # #    --- 可视化逻辑 ---
                # if args.local_rank == 0:
                #     cam_vis_dir = os.path.join(base_dir, "Train_Low_iter0")
                #     os.makedirs(cam_vis_dir, exist_ok=True)

                #     present_fgs = []
                #     for i in range(cls_label.shape[1]): # 自动适应 size 20
                #         if cls_label[0, i] > 0:
                #             present_fgs.append(i + 1) # 假设 valid_cam 的 0 是背景，1-20 是物体

                #     if len(present_fgs) > 0:
                #         # 2. 像素级取最大值融合
                #         fg_cams = valid_cam[0, present_fgs] # [num_present, H, W]
                #         combined_cam, _ = torch.max(fg_cams, dim=0)
                        
                #         # 3. 后续处理保持不变
                #         combined_cam = F.interpolate(combined_cam.unsqueeze(0).unsqueeze(0), 
                #                                 size=(H, W), mode='bilinear').squeeze()
                #         cam_np = combined_cam.cpu().numpy()
                #         cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
                        
                #         import cv2 # 确保开头导入了 cv2
                #         cam_uint8 = np.uint8(255 * cam_np)
                #         heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
                        
                #         save_path = os.path.join(cam_vis_dir, f"{name[0]}.png")
                #         cv2.imwrite(save_path, heatmap)

                if args.use_tta:
                    flipped_inputs = torch.flip(scaled_inputs, dims=[3])
                    flip_pred = model(flipped_inputs, cls_label, labels)
                    segs_flip = torch.flip(flip_pred, dims=[3])
                    segs_flip = F.interpolate(segs_flip, size=(H, W), mode='bilinear', align_corners=False)
                    preds_logit_list.append(segs_flip)

            avg_logits = torch.mean(torch.stack(preds_logit_list), dim=0)
            
            # 获取 Softmax  (用于 CRF)
            probs = torch.softmax(avg_logits, dim=1).squeeze().cpu().numpy() # [C, H, W]

            # ========== DenseCRF ==========
            if args.use_crf:
                img_temp = F.interpolate(inputs, size=(H, W), mode='bilinear', align_corners=False)
                img_temp = img_temp.squeeze().cpu().numpy()
                img_temp = np.transpose(img_temp, (1, 2, 0)) # [H, W, 3]
                
                img_temp = img_temp * 255.0
                img_temp = img_temp.astype(np.uint8)
                
                crf_probs = crf_inference(
                    img=img_temp, 
                    probs=probs, 
                    t=10,           # 迭代次数，10次通常比5次更精细
                    scale_factor=1, 
                    labels=args.num_classes
                )
                pred_mask = np.argmax(crf_probs, axis=0) # [H, W]
            else:
                # 直接取 argmax
                pred_mask = np.argmax(probs, axis=0)

            gt_mask = labels.squeeze().to(torch.uint8).cpu().numpy()
           
            preds.append(pred_mask)
            gts.append(gt_mask)
            # _pseudo_labels.append(attn_label[0].cpu().numpy().astype(np.int16))

            # if args.local_rank == 0: # 仅主进程保存

            #     img_name = name[0]

            #     img_path = os.path.join(args.data_folder, 'JPEGImages', img_name + '.jpg')
            #     img_pil = Image.open(img_path).convert('RGB')
            #     width, height = img_pil.size

            #     gt_img = decode_segmap(gt_mask).resize((width, height), Image.NEAREST)
            #     pred_img = decode_segmap(pred_mask).resize((width, height), Image.NEAREST)
            #     attn_img = decode_segmap(attn_label[0].cpu().numpy().astype(np.int16)).resize((width, height), Image.NEAREST)
          
            #     # 2. 拼接图片 (原图 | GT | 预测结果 | 伪标签 )
            #     combined_img = Image.new('RGB', (width * 1, height))

            #     # combined_img.paste(img_pil, (0, 0))
            #     # combined_img.paste(gt_img, (width, 0))
            #     combined_img.paste(pred_img, (0, 0))     
            #     # combined_img.paste(attn_img, (0, 0))     

            #     save_path = os.path.join(pred_dir, f"{img_name}.png")
            #     combined_img.save(save_path)

          
    seg_score = evaluate.scores(gts, preds)
    tab_results = format_tabs([seg_score], name_list=["Pred"], cat_list=voc.class_list)

    if args.local_rank == 0:
        print(f"Results: {tab_results}")

        precision, recall = compute_precision_recall_scores(gts, preds, args.num_classes, args.ignore_index)

        print("\n" + "=" * 65)
        print(f"{'Class Name':<20} | {'Precision':<15} | {'Recall':<15}")
        print("-" * 65)
        
        # 假设 voc.class_list 包含所有类别名称（含背景）
        # 如果 voc.class_list 长度与 precision 不一致，请注意调整索引
        class_names = voc.class_list 
        
        mPrec_list = []
        mRec_list = []

        for i in range(args.num_classes):
            # 获取类别名称，防止越界
            c_name = class_names[i] if i < len(class_names) else f"Class {i}"
            
            p_val = precision[i] * 100
            r_val = recall[i] * 100
            
            print(f"{c_name:<20} | {p_val:>10.2f} %    | {r_val:>10.2f} %")
            
            mPrec_list.append(precision[i])
            mRec_list.append(recall[i])

        print("-" * 65)
        print(f"{'Mean (All)':<20} | {np.mean(mPrec_list)*100:>10.2f} %    | {np.mean(mRec_list)*100:>10.2f} %")
        print("=" * 65 + "\n")
        # ==========================================================================

    return tab_results  # 修正返回值


if __name__ == "__main__":
    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    setup_seed(args.seed)

    args.use_tta = True
    args.use_crf = False

    tab_results = validate(args=args)

'''
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/test_voc.py \
    --work_dir work_dir_voc
'''