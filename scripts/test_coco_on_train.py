import argparse
import datetime
import logging
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil

from datasets import coco as coco
from model.model_seg_neg import DiffusionBasedNetwork
from utils.pyutils import format_tabs

torch.hub.set_dir("./pretrained")

parser = argparse.ArgumentParser()

parser.add_argument("--diffusion_model", default='stabilityai/stable-diffusion-2-1-base', type=str, help="Diffusion model name")
parser.add_argument("--img_folder", default='/media/store1/xln/DiffusionCAM/MSCOCO/coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='/media/store1/xln/DiffusionCAM/MSCOCO/SegmentationClass', type=str, help="label folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="ignore index")

parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work dir")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split (unused here)")
parser.add_argument("--spg", default=1, type=int, help="samples per gpu")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weight decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="power factor for poly scheduler")

parser.add_argument("--max_iters", default=160000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help="logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=16000, type=int, help="warmup iters")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save ckpt")

parser.add_argument("--local_rank", default=-1, type=int, help="local rank")
parser.add_argument("--num_workers", default=8, type=int, help="num workers")
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

parser.add_argument("--use_tta", action="store_true", default=True, help="use multi-scale TTA")
parser.add_argument("--use_crf", action="store_true", default=True, help="use DenseCRF")
parser.add_argument("--memory_log_interval", default=100, type=int, help="print RAM usage every N iters")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_coco_palette(num_classes=256):
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
    palette = get_coco_palette(256)
    rgb = palette[label_mask]
    if ignore_index is not None:
        rgb[label_mask == ignore_index] = [255, 255, 255]
    return Image.fromarray(rgb)


def fast_hist(label_true, label_pred, num_classes, ignore_index=255):
    """
    在线累计混淆矩阵，避免把所有 mask 存到内存里。
    """
    label_true = label_true.astype(np.int64)
    label_pred = label_pred.astype(np.int64)

    mask = (label_true != ignore_index) & (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask] + label_pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).astype(np.float64)
    return hist


def scores_from_hist(hist):
    """
    模拟常见语义分割评估指标
    """
    eps = 1e-10
    acc = np.diag(hist).sum() / (hist.sum() + eps)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + eps)
    mean_acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + eps)
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / (hist.sum() + eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {
        "pAcc": acc * 100.0,
        "mAcc": mean_acc_cls * 100.0,
        "miou": mean_iu * 100.0,
        "fwIoU": fwavacc * 100.0,
        "iou": iu * 100.0,
    }


def validate(args=None):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    args.local_rank = local_rank

    world_size = dist.get_world_size()
    if args.local_rank == 0:
        print(f"Total gpus: {world_size}, samples per gpu: {args.spg}")

    time0 = datetime.datetime.now().replace(microsecond=0)

    # 这里按你的要求，仍然跑 train split，不改成 args.val_set
    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
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

    device = torch.device(f"cuda:{args.local_rank}")

    model = DiffusionBasedNetwork(
        num_classes=args.num_classes,
        backbone=args.diffusion_model,
        attention_layers_to_use=args.attention_layers_to_use,
        dataset_name='coco2014',
        cam_bg_thr=0.45,
        ent=0.1,
    )
    model = model.to(device)

    checkpoint_path = 'work_dir_coco/wo_Dr/checkpoints/model_iter_160000.pth'

    if args.local_rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
        print(f"Configurations -> TTA: {args.use_tta}, CRF: {args.use_crf}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    msg = model.load_state_dict(new_state_dict, strict=False)
    if args.local_rank == 0:
        print("load_state_dict:", msg)

    model.eval()

    # 不再保存所有样本结果，改为在线统计
    seg_hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    # pseudo_hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)

    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    pred_dir = os.path.join(base_dir, "Pred_Train", args.train_set)
    os.makedirs(pred_dir, exist_ok=True)

    scales = [1.0, 1.5, 1.2] if args.use_tta else [1.0]
    process = psutil.Process(os.getpid())

    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            ncols=100,
            ascii=" >=",
            desc='Validating',
            disable=args.local_rank != 0
        ):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            cls_label = cls_label.cuda(non_blocking=True)

            H, W = labels.shape[1], labels.shape[2]
            preds_logit_list = []

            # refined_label_last = None

            for scale in scales:
                if scale != 1.0:
                    scaled_inputs = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    scaled_inputs = inputs

                # pred, refined_label = model(scaled_inputs, cls_label, labels)
                pseudo_logits = model(scaled_inputs, cls_label, labels)
                # refined_label_last = refined_label

                segs = F.interpolate(pseudo_logits, size=(H, W), mode='bilinear', align_corners=False)
                preds_logit_list.append(segs)

                if args.use_tta:
                    flipped_inputs = torch.flip(scaled_inputs, dims=[3])
                    flip_pred = model(flipped_inputs, cls_label, labels)
                    segs_flip = torch.flip(flip_pred, dims=[3])
                    segs_flip = F.interpolate(segs_flip, size=(H, W), mode='bilinear', align_corners=False)
                    preds_logit_list.append(segs_flip)

            avg_logits = torch.mean(torch.stack(preds_logit_list, dim=0), dim=0)

            # 预测分割图
            pred_mask = torch.argmax(avg_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # GT
            gt_mask = labels.squeeze(0).cpu().numpy().astype(np.uint8)

            # # 伪标签
            # label_mask = refined_label_last[0].cpu().numpy().astype(np.uint8)

            # 在线累计混淆矩阵
            seg_hist += fast_hist(gt_mask, pred_mask, args.num_classes, args.ignore_index)
            # pseudo_hist += fast_hist(gt_mask, label_mask, args.num_classes, args.ignore_index)

            # 定期打印内存情况
            if args.local_rank == 0 and (idx + 1) % args.memory_log_interval == 0:
                ram_gb = process.memory_info().rss / 1024**3
                gpu_mem_mb = torch.cuda.memory_allocated(device) / 1024**2
                print(f"[Iter {idx+1}/{len(val_loader)}] RAM={ram_gb:.2f} GB, GPU_mem={gpu_mem_mb:.2f} MB")

    # 多卡时聚合混淆矩阵
    seg_hist_tensor = torch.from_numpy(seg_hist).to(device)
    # pseudo_hist_tensor = torch.from_numpy(pseudo_hist).to(device)

    dist.all_reduce(seg_hist_tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(pseudo_hist_tensor, op=dist.ReduceOp.SUM)

    seg_hist = seg_hist_tensor.cpu().numpy()
    # pseudo_hist = pseudo_hist_tensor.cpu().numpy()

    seg_score = scores_from_hist(seg_hist)
    # pseudo_score = scores_from_hist(pseudo_hist)

    if args.local_rank == 0:
        print("\n========== Final Results ==========")
        print(f"Pred   mIoU: {seg_score['miou']:.4f}")
        # print(f"Pseudo mIoU: {pseudo_score['miou']:.4f}")
        print(f"Pred   pAcc: {seg_score['pAcc']:.4f}")
        # print(f"Pseudo pAcc: {pseudo_score['pAcc']:.4f}")

        # 如果你还想看每类 IoU，可以打开下面注释
        print("\nPer-class IoU (Pred):")
        for i, cls_name in enumerate(coco.class_list):
            print(f"{i:02d} {cls_name:<20}: {seg_score['iou'][i]:.4f}")

        # print("\nPer-class IoU (Pseudo):")
        # for i, cls_name in enumerate(coco.class_list):
        #     print(f"{i:02d} {cls_name:<20}: {pseudo_score['iou'][i]:.4f}")

    dist.barrier()
    dist.destroy_process_group()

    # return seg_score, pseudo_score
    return seg_score


if __name__ == "__main__":
    args = parser.parse_args()

    setup_seed(args.seed)

    # 按你当前需求，这里先关掉 TTA / CRF
    args.use_tta = False
    args.use_crf = False

    # seg_score, pseudo_score = validate(args=args)
    seg_score = validate(args=args)

"""
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    scripts/test_coco_on_train.py \
    --work_dir work_dir_coco
"""