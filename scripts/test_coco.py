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
from datasets import coco as coco
warnings.filterwarnings('ignore')

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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

parser.add_argument("--diffusion_model", default='stabilityai/stable-diffusion-2-1-base', type=str, help="Diffusion model name")
parser.add_argument("--img_folder", default='/media/store1/xln/DiffusionCAM/MSCOCO/coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/media/store1/xln/DiffusionCAM/MSCOCO/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work_dir_coco_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--spg", default=1, type=int, help="samples_per_gpu")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="power factor for poly scheduler")

parser.add_argument("--max_iters", default=160000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=16000, type=int, help="warmup_iters")

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


def get_coco_palette(num_classes=256):
    """
    生成标准的 COCO/PASCAL VOC 风格调色板。
    为了防止 ignore_index (255) 越界，通常直接生成 256 个颜色。
    """
    n = num_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    
    return np.array(palette).reshape(-1, 3).astype(np.uint8)

def decode_segmap(label_mask, num_classes=81, ignore_index=255):
    """
    将 Segmentation Mask 转换为 RGB 图像 (COCO版)
    """
    # 1. 获取调色板 (生成 256 个颜色以覆盖所有可能的 ID，包括 ignore_index)
    palette = get_coco_palette(256)
 
    rgb = palette[label_mask]
  
    if ignore_index is not None:
        rgb[label_mask == ignore_index] = [255, 255, 255]

    return Image.fromarray(rgb)


def validate(args=None):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    args.local_rank = local_rank
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split= args.train_set, 
        stage='train',
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
        dataset_name='coco2014',
        cam_bg_thr=0.45,
        ent=0.1,
    )

    model = model.to(device)
    checkpoint_path = 'work_dir_coco/dino+affinity0.1/checkpoints/model_iter_156000.pth'
    
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
    
    scales = [1.0, 1.5, 1.2] if args.use_tta else [1.0]

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
                
                pred, refined_label = model(scaled_inputs, cls_label, labels)
                # _, _, pred, _ = model(inputs, cls_label, labels)
                
                segs = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
                preds_logit_list.append(segs)

                if args.use_tta:
                    flipped_inputs = torch.flip(scaled_inputs, dims=[3])
                    flip_pred, flip_refined_label = model(flipped_inputs, cls_label, labels)
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
            label_mask = refined_label[0].cpu().numpy().astype(np.int16)

            preds.append(pred_mask)
            gts.append(gt_mask)
            _pseudo_labels.append(label_mask)
          
            # if args.local_rank == 0: # 仅主进程保存

            #     img_name = name[0]

            #     img_path = os.path.join(args.img_folder, 'train2014', img_name + '.jpg')
            #     img_pil = Image.open(img_path).convert('RGB')
            #     width, height = img_pil.size

            #     gt_img = decode_segmap(gt_mask).resize((width, height), Image.NEAREST)
            #     pred_img = decode_segmap(pred_mask).resize((width, height), Image.NEAREST)
          
            #     # 2. 拼接图片 (原图 | GT | 预测结果)
            #     combined_img = Image.new('RGB', (width * 3, height))

            #     combined_img.paste(img_pil, (0, 0))
            #     combined_img.paste(gt_img, (width, 0))
            #     combined_img.paste(pred_img, (width * 2, 0))    

            #     save_path = os.path.join(pred_dir, f"{img_name}.png")
            #     combined_img.save(save_path)

    seg_score = evaluate.scores(gts, preds, args.num_classes)
    label_score = evaluate.scores(gts, _pseudo_labels, args.num_classes)

    tab_results = format_tabs([seg_score, label_score], name_list=["Pred", "Mask" ], cat_list=coco.class_list)
    
    if args.local_rank == 0:
        print(f"Results: {tab_results}")
        
    return tab_results  # 修正返回值


if __name__ == "__main__":
    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    setup_seed(args.seed)

    args.use_tta = False
    args.use_crf = False

    tab_results = validate(args=args)

'''
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/test_coco.py \
    --work_dir work_dir_coco
'''