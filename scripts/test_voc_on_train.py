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
parser.add_argument("--diffusion_model", default='Manojb/stable-diffusion-2-1-base', type=str)
parser.add_argument("--data_folder", default='/media/store2/xln/FIDi/VOCdevkit/VOC2012', type=str)
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
    checkpoint_path = 'work_dir_voc/warm_up=8000/checkpoints/model_iter_46000.pth'
    
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

    # 定义两个列表，分别存储模型预测和Refined伪标签
    preds_model = []
    preds_refined = []
    gts = []
    
    model.eval()
    
    scales = [1.0, 1.5, 1.25] if args.use_tta else [1.0]

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >=", 
                            desc=f'Evaluating {args.train_set}', disable=args.local_rank != 0):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            H, W = labels.shape[1], labels.shape[2]

            logit_list_model = []    # 存储 Model Logits
            prob_list_refined = []   # 存储 Refined One-Hot
            
            for scale in scales:
                # ---------------- TTA Scale ----------------
                if scale != 1.0:
                    scaled_inputs = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    scaled_inputs = inputs
                
                # 获取输出
                refined_label = model(scaled_inputs, cls_label, labels)
                # pseudo_logits, attn_label, refine_label = model(inputs, cls_label, mask)
                
                # --- 1. 处理 Model Prediction ---
                # segs = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                # logit_list_model.append(segs)

                # --- 2. 处理 Refined Label (One-Hot Voting) ---
                mask_temp = refined_label.clone().long()
                
                # 处理 255 (Ignore Index)
                mask_valid = (mask_temp != 255)
                mask_temp[~mask_valid] = 0 # 临时置0，防止one_hot报错
                
                # 转 One-Hot: [1, H, W] -> [1, H, W, C] -> [1, C, H, W]
                mask_onehot = F.one_hot(mask_temp, num_classes=args.num_classes).float()
                mask_onehot = mask_onehot.permute(0, 3, 1, 2)
                
                # (可选) 如果你需要严格忽略 255 区域的投票，可以在这里把对应位置概率全置0
                # mask_onehot[~mask_valid.unsqueeze(1).expand_as(mask_onehot)] = 0

                refined = F.interpolate(mask_onehot, size=(H, W), mode='bilinear', align_corners=False)
                prob_list_refined.append(refined)

                # ---------------- TTA Flip ----------------
                if args.use_tta:
                    flipped_inputs = torch.flip(scaled_inputs, dims=[3])
                    refined_label_flip = model(flipped_inputs, cls_label, labels)
                    
                    # 1. Flip Model
                    # segs_flip = torch.flip(flip_pred, dims=[3])
                    # segs_flip = F.interpolate(segs_flip, size=(H, W), mode='bilinear', align_corners=False)
                    # logit_list_model.append(segs_flip)

                    # 2. Flip Refined
                    mask_temp_flip = refined_label_flip.clone().long()
                    mask_valid_flip = (mask_temp_flip != 255)
                    mask_temp_flip[~mask_valid_flip] = 0
                    
                    mask_onehot_flip = F.one_hot(mask_temp_flip, num_classes=args.num_classes).float()
                    mask_onehot_flip = mask_onehot_flip.permute(0, 3, 1, 2)
                    
                    # 翻转回来
                    mask_onehot_flip = torch.flip(mask_onehot_flip, dims=[3]) 
                    segs_flip_refined = F.interpolate(mask_onehot_flip, size=(H, W), mode='bilinear', align_corners=False)
                    prob_list_refined.append(segs_flip_refined)

            # === 聚合 Model Pred ===
            # avg_logits = torch.mean(torch.stack(logit_list_model), dim=0)
            
            # if args.use_crf:
            #     # CRF 需要概率图
            #     # probs = torch.softmax(avg_logits, dim=1).squeeze().cpu().numpy()
            #     img_temp = F.interpolate(inputs, size=(H, W), mode='bilinear', align_corners=False)
            #     img_temp = img_temp.squeeze().cpu().numpy().transpose(1, 2, 0)
            #     img_temp = (img_temp * 255.0).astype(np.uint8)
                
            #     # crf_probs = crf_inference(img_temp, probs, t=10, scale_factor=1, labels=args.num_classes)
            #     # pred_mask = np.argmax(crf_probs, axis=0)
            # else:
            #     # pred_mask = torch.argmax(avg_logits, dim=1).squeeze().cpu().numpy()

            # preds_model.append(pred_mask)

            # === 聚合 Refined Label ===
            # 【重要】这里得到的 avg_refined 已经是平均后的 One-Hot 向量，直接 Argmax 即可
            
            avg_refined = torch.mean(torch.stack(prob_list_refined), dim=0)
            refined_mask = torch.argmax(avg_refined, dim=1).squeeze().cpu().numpy() # [H, W]
            
            preds_refined.append(refined_mask) # 存入 Mask 而不是 Probability

            # === GT ===
            gt_mask = labels.squeeze().to(torch.uint8).cpu().numpy()
            gts.append(gt_mask)

    # 1. 计算 mIoU
    # seg_score = evaluate.scores(gts, preds_model)
    refined_score = evaluate.scores(gts, preds_refined)

    # 2. 打印 IoU 表格
    # tab_results = format_tabs([seg_score, refined_score], name_list=["Model_Pred", "Refined_Mask"], cat_list=voc.class_list)
    tab_results = format_tabs([refined_score], name_list=["Refined_Mask"], cat_list=voc.class_list)
    
    if args.local_rank == 0:
        print("\n" + "#" * 60)
        print(">>> mIoU Results")
        print(tab_results)

    #     # 3. 计算并打印 Precision / Recall (Model Pred)
    #     prec_model, rec_model = compute_precision_recall_scores(gts, preds_model, args.num_classes, args.ignore_index)
    #     print_pr_table(prec_model, rec_model, voc.class_list, title="Model Pred P/R")

    #     # 4. 计算并打印 Precision / Recall (Refined Mask)
    #     prec_refined, rec_refined = compute_precision_recall_scores(gts, preds_refined, args.num_classes, args.ignore_index)
    #     print_pr_table(prec_refined, rec_refined, voc.class_list, title="Refined Mask P/R")
        
    return tab_results


if __name__ == "__main__":
    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    setup_seed(args.seed)

    # 强制开启 TTA
    args.use_tta = True
    args.use_crf = False
    
    validate(args=args)

'''
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/test_voc_on_train.py \
    --work_dir work_dir_voc
'''