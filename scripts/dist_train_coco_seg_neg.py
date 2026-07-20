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
from datasets import coco as coco
from model.losses import get_seg_loss, OHEMCrossEntropyLoss
from model.model_seg_neg import DiffusionBasedNetwork
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.cam import PAR
from utils import evaluate, imutils, optimizer
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
parser.add_argument("--exp", required=True, type=str, help="exp_name")

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None):

    Preds, Masks, GTs = [], [], []
    model.eval()
    avg_meter = AverageMeter()

    with torch.no_grad():
        # 只在主进程显示验证进度条
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >=", 
                           desc='Validating', disable=args.local_rank != 0):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()  
            cls_label = cls_label.cuda()  

            # pseudo_logits, attn_label, pred, refine_label = model(inputs, cls_label, labels)
            # _, _, pred, refine_label = model(inputs, cls_label, labels)
            pseudo_logits, _ = model.module(inputs, cls_label, labels)  # 直接访问底层模型
            
            # pred = F.interpolate(pred, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            pred = F.interpolate(pseudo_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            Preds.append(torch.argmax(pred, dim=1).cpu().numpy().astype(np.int16))
            # Masks.append(attn_label[0].cpu().numpy().astype(np.int16))
            # Masks.append(refine_label[0].cpu().numpy().astype(np.int16))
            GTs.append(labels[0].cpu().numpy().astype(np.int16))
            
    pred_score = evaluate.scores(GTs, Preds, args.num_classes)
    # mask_score = evaluate.scores(GTs, Masks, args.num_classes)
    model.train()

    tab_results = format_tabs([pred_score], name_list=["Pred"], cat_list=coco.class_list)
    # tab_results = format_tabs([pred_score, mask_score], name_list=["Pred", "Mask"], cat_list=coco.class_list)

    return tab_results
 
def train(args=None):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    args.local_rank = local_rank
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        rescale=None,
        hor_flip=False,
        crop_method='interpolate'
    )

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        rescale=None,
        hor_flip=False,
        crop_method='interpolate'
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
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
    param_groups = model.get_param_groups()
    model.to(device)

    optim = getattr(optimizer, args.optimizer)(
       params=[
            {
                "params": param_groups[0],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            }
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters, 
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    ohem_loss_fn = OHEMCrossEntropyLoss(ignore_index=255, min_kept=65000)

    def get_ohem_loss(pred, label, ignore_index=255):
        ohem_loss_fn.ignore_index = ignore_index
        ohem_loss_fn.criterion.ignore_index = ignore_index
        return ohem_loss_fn(pred, label)

    # 使用tqdm创建进度条
    pbar = tqdm(range(args.max_iters), desc='Training', ncols=100, ascii=' >=', 
                disable=args.local_rank != 0)  # 只在主进程显示进度条
    
    for n_iter in pbar:
        try:
            img_name, inputs, mask, cls_label = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)         
            img_name, inputs, mask, cls_label = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        cls_label = cls_label.to(device, non_blocking=True)
        
        # pseudo_logits, attn_label, pred, refine_label = model(inputs, cls_label, mask)

        pseudo_logits, attn_label = model(inputs, cls_label, mask)


        label_loss = get_ohem_loss(pseudo_logits, attn_label.type(torch.long), ignore_index=args.ignore_index)
        # pred_loss = get_ohem_loss(pred, refine_label.type(torch.long), ignore_index=args.ignore_index)

        # warmup
        # if n_iter <= args.warmup_iters:
        #     w_pred = 1.0 * (n_iter + 1) / args.warmup_iters
        # else:
        #     w_pred = 1.0
        
        # loss = 1.0 * label_loss + w_pred * pred_loss
        loss = 1.0 * label_loss

        avg_meter.add({
            'label_loss': label_loss.item(),
            # 'pred_loss': pred_loss.item(),
        })

        # 每次迭代都更新进度条
        if args.local_rank == 0:
            pbar.set_postfix({
                'label_loss': f'{label_loss.item():.4f}',
                # 'pred_loss': f'{pred_loss.item():.4f}'
            })

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            # if args.local_rank == 0:
            #     logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; label_loss: %.4f; pred_loss: %.4f" % 
            #                (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('label_loss'), avg_meter.pop('pred_loss')))
            
            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; label_loss: %.4f" % 
                           (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('label_loss')))

        if (n_iter + 1) >= 90000 and (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            tab_results = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("\n" + tab_results)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    args.work_dir = os.path.join(args.work_dir, args.exp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    setup_seed(args.seed)
    train(args=args)
