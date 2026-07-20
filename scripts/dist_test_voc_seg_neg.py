import argparse
import datetime
import logging
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(".")

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import argparse
import datetime
import logging
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import imageio
from datasets import voc as voc
from model.losses import get_seg_loss, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg import DiffusionBasedNetwork
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.cam import PAR
from utils import evaluate, imutils, optimizer
from utils.dcrf import crf_inference
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
import matplotlib.pyplot as plt
torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

# parser.add_argument("--diffusion_model", default='stable-diffusion-v1-5/stable-diffusion-v1-5', type=str, help="Diffusion model name")
parser.add_argument("--diffusion_model", default='Manojb/stable-diffusion-2-1-base', type=str, help="Diffusion model name")
parser.add_argument("--data_folder", default='/media/store2/xln/FIDi/VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str, help="work_dir_voc_wseg")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=1, type=int, help="samples_per_gpu")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="power factor for poly scheduler")

parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--w_seg", default=0.5, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.1, type=float, help="w_reg")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")  
parser.add_argument('--backend', default='nccl')

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

parser.add_argument("--use_tta", action="store_true", default=True, help="是否使用多尺度TTA")
parser.add_argument("--use_crf", action="store_true", default=True, help="是否使用DenseCRF后处理")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def choose_vis_class(mask_np, cls_label):
    # 优先用 pseudo mask 中的前景类
    fg_classes = np.unique(mask_np)
    fg_classes = fg_classes[(fg_classes > 0) & (fg_classes <= 20)]

    if len(fg_classes) > 0:
        return int(fg_classes[0])
    else:
        return int(torch.argmax(cls_label[0, 1:]).item() + 1)


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

def compute_precision_recall_scores(gts, preds, num_classes, ignore_index=255):
    """
    通过计算混淆矩阵来获取每类的 Precision 和 Recall
    """
    hist = np.zeros((num_classes, num_classes))
    
    print("Calculating Precision & Recall...")
    for gt, pred in zip(gts, preds):
        # 展平并过滤掉 ignore_index
        gt = gt.flatten()
        pred = pred.flatten()
        mask = (gt != ignore_index)
        
        gt_valid = gt[mask].astype(int)
        pred_valid = pred[mask].astype(int)
        
        # 快速计算混淆矩阵
        # 技巧：将二维坐标(gt, pred)映射为一维索引 (gt * num_classes + pred)
        count = np.bincount(
            num_classes * gt_valid + pred_valid,
            minlength=num_classes ** 2
        )
        hist += count.reshape(num_classes, num_classes)

    # TP: 对角线元素
    tp = np.diag(hist)
    # TP + FP: 预测为该类的总数 (列求和)
    sum_pred = hist.sum(axis=0)
    # TP + FN: 真实为该类的总数 (行求和)
    sum_gt = hist.sum(axis=1)

    # Precision = TP / (TP + FP)
    precision = np.divide(tp, sum_pred, out=np.zeros_like(tp), where=sum_pred!=0)
    
    # Recall = TP / (TP + FN)
    recall = np.divide(tp, sum_gt, out=np.zeros_like(tp), where=sum_gt!=0)

    return precision, recall

def validate(args=None):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    args.local_rank = local_rank
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
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
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 处理DDP模型的键名
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
    avg_meter = AverageMeter()

    # 创建CAM保存目录
    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    pred_dir = os.path.join(base_dir, "Pred", args.val_set)  
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
                
                pred = model(scaled_inputs, cls_label, labels)
                # _, _, pred, _ = model(inputs, cls_label, labels)
                
                segs = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                preds_logit_list.append(segs)

                # --- 可视化逻辑 ---
                # if args.local_rank == 0:
                #     cam_vis_dir = os.path.join(base_dir, "Low+High+DR_Test")
                #     os.makedirs(cam_vis_dir, exist_ok=True)

                #     present_fgs = []
                #     for i in range(cls_label.shape[1]): # 自动适应 size 20
                #         if cls_label[0, i] > 0:
                #             present_fgs.append(i + 1) # 假设 valid_cam 的 0 是背景，1-20 是物体

                #     if len(present_fgs) > 0:
                #         # 2. 像素级取最大值融合
                #         fg_cams = refined_pseudo_logits[0, present_fgs] # [num_present, H, W]
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

            # if args.local_rank == 0: # 仅主进程保存

            #     img_name = name[0]

            #     img_path = os.path.join(args.data_folder, 'JPEGImages', img_name + '.jpg')
            #     img_pil = Image.open(img_path).convert('RGB')
            #     width, height = img_pil.size

            #     gt_img = decode_segmap(gt_mask).resize((width, height), Image.NEAREST)
            #     pred_img = decode_segmap(pred_mask).resize((width, height), Image.NEAREST)
          
            #     # 2. 拼接图片 (原图 | GT | 预测结果 | 伪标签 )
            #     combined_img = Image.new('RGB', (width * 3, height))

            #     combined_img.paste(img_pil, (0, 0))
            #     combined_img.paste(gt_img, (width, 0))
            #     combined_img.paste(pred_img, (width * 2, 0))     

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
    scripts/dist_test_voc_seg_neg.py \
    --work_dir work_dir_voc
'''