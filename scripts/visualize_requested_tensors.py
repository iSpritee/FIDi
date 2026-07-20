import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")

from datasets import voc as voc  # noqa: E402
from model.model_seg_neg import DiffusionBasedNetwork  # noqa: E402


DEFAULT_ATTENTION_LAYERS = [
    "down_blocks[1].attentions[1].transformer_blocks[0].attn2",
    "down_blocks[2].attentions[0].transformer_blocks[0].attn2",
    "down_blocks[2].attentions[1].transformer_blocks[0].attn2",
    "up_blocks[1].attentions[0].transformer_blocks[0].attn2",
    "up_blocks[1].attentions[1].transformer_blocks[0].attn2",
    "up_blocks[1].attentions[2].transformer_blocks[0].attn1",
    "up_blocks[1].attentions[2].transformer_blocks[0].attn2",
    "up_blocks[2].attentions[0].transformer_blocks[0].attn2",
    "up_blocks[2].attentions[1].transformer_blocks[0].attn2",
    "up_blocks[3].attentions[1].transformer_blocks[0].attn1",
    "mid_block.attentions[0].transformer_blocks[0].attn1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize att_map, fused_feature, pseudo_logits_low, and block11 with spectra."
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir_voc/t=200/checkpoints/model_iter_46000.pth",
        type=str,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--diffusion_model",
        default="stabilityai/stable-diffusion-2-1-base",
        type=str,
        help="Diffusion backbone name.",
    )
    parser.add_argument(
        "--feature_extractor",
        default="facebook/dinov2-with-registers-base",
        type=str,
        help="DINO backbone name.",
    )
    parser.add_argument(
        "--data_folder",
        default="/media/store1/xln/DiffusionCAM/VOCdevkit/VOC2012",
        type=str,
        help="VOC dataset root.",
    )
    parser.add_argument(
        "--list_folder",
        default="datasets/voc",
        type=str,
        help="VOC split list folder.",
    )
    parser.add_argument(
        "--split",
        default="val",
        type=str,
        help="Dataset split, e.g. val or train_aug.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        type=str,
        help="Dataset stage. Defaults to train for train splits, otherwise val.",
    )
    parser.add_argument(
        "--sample_index",
        default=0,
        type=int,
        help="Sample index when sample_name is not provided.",
    )
    parser.add_argument(
        "--sample_name",
        default=None,
        type=str,
        help="VOC image id, e.g. 2007_000032.",
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations/requested_tensors",
        type=str,
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--num_classes",
        default=21,
        type=int,
        help="Number of segmentation classes including background.",
    )
    parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=DEFAULT_ATTENTION_LAYERS,
        help="Attention layers used by the extractor hooks.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_stage(split: str, stage: str = None) -> str:
    if stage is not None:
        return stage
    return "train" if "train" in split else "val"


def build_dataset(args):
    return voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.split,
        stage=get_stage(args.split, args.stage),
        rescale=None,
        hor_flip=False,
        crop_method="interpolate",
    )


def find_sample_index(dataset, sample_name: str) -> int:
    for idx, name in enumerate(dataset.name_list):
        if name == sample_name:
            return idx
    raise ValueError(f"sample_name '{sample_name}' not found in split '{dataset.name_list_dir}'.")


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def tensor_to_chw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for visualization, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Unsupported tensor shape for visualization: {tuple(tensor.shape)}")
    return tensor.detach().cpu().float()


def feature_spatial_summary(tensor: torch.Tensor, mode: str) -> np.ndarray:
    chw = tensor_to_chw(tensor)
    if mode == "attention_mean":
        summary = chw.mean(dim=0)
    elif mode == "feature_l2":
        summary = torch.norm(chw, p=2, dim=0)
    elif mode == "logit_confidence":
        probs = torch.softmax(chw, dim=0)
        summary = probs.max(dim=0).values
    else:
        raise ValueError(f"Unknown summary mode: {mode}")
    return summary.numpy()


def mean_log_power_spectrum(tensor: torch.Tensor) -> np.ndarray:
    chw = tensor_to_chw(tensor)
    chw = chw - chw.mean(dim=(-2, -1), keepdim=True)
    channels = chw.numpy().astype(np.float32)

    power_maps = []
    for channel in channels:
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)
        power_maps.append(np.abs(fft_shift) ** 2)

    mean_power = np.mean(np.stack(power_maps, axis=0), axis=0)
    return np.log1p(mean_power)


def save_heatmap(np_map: np.ndarray, path: str, cmap: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np_map, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_panel(spatial_map: np.ndarray, spectrum_map: np.ndarray, path: str, title: str, spatial_cmap: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    im0 = axes[0].imshow(normalize_map(spatial_map), cmap=spatial_cmap)
    axes[0].set_title("Spatial Summary", fontsize=13)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(normalize_map(spectrum_map), cmap="inferno")
    axes[1].set_title("Mean Log Power Spectrum", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_channel_grid(
    tensor: torch.Tensor,
    path: str,
    title: str,
    channel_names: Sequence[str],
    cmap: str = "jet",
    max_cols: int = 4,
):
    chw = tensor_to_chw(tensor)
    num_channels = chw.shape[0]
    num_cols = min(max_cols, num_channels)
    num_rows = int(np.ceil(num_channels / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 4.2 * num_rows))
    fig.suptitle(title, fontsize=16)

    axes = np.array(axes).reshape(num_rows, num_cols)
    for idx in range(num_rows * num_cols):
        ax = axes[idx // num_cols, idx % num_cols]
        if idx >= num_channels:
            ax.axis("off")
            continue
        channel = normalize_map(chw[idx].numpy())
        im = ax.imshow(channel, cmap=cmap)
        name = channel_names[idx] if idx < len(channel_names) else f"channel_{idx}"
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def denormalize_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor[0].detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = strip_module_prefix(checkpoint)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def fuse_with_existing_head(fuser, input_image: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
    projected_features = []
    for feat, projector in zip(features, fuser.projections):
        projected_features.append(projector(feat))

    target_h = max(feat.shape[2] for feat in projected_features)
    target_w = max(feat.shape[3] for feat in projected_features)

    resized_features = []
    for feat in projected_features:
        if feat.shape[2:] != (target_h, target_w):
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        resized_features.append(feat)

    input_image_resized = F.interpolate(
        input_image,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(input_image.device)
    resized_features.append(input_image_resized)

    fused_with_rgb = torch.cat(resized_features, dim=1)
    return fuser.final_conv(fused_with_rgb)


@torch.no_grad()
def extract_requested_tensors(model: DiffusionBasedNetwork, image: torch.Tensor, cls_label: torch.Tensor):
    active_classes = torch.where(cls_label[0] == 1)[0]
    if len(active_classes) == 0:
        raise RuntimeError("The selected sample has no active foreground classes.")

    text_embedding = model.prepare_text_embeddings(active_classes)
    all_features, cross_attention_maps, self_attention_maps = model.fuser.extractor(image, text_embedding)
    att_map = model.get_att_map(cross_attention_maps, self_attention_maps)
    fused_feature = fuse_with_existing_head(model.fuser, image, all_features)
    pseudo_logits_low = model.pseudo_mask_generator(fused_feature)
    block11_feature = all_features[-1]

    return {
        "active_classes": active_classes,
        "all_features": all_features,
        "att_map": att_map,
        "fused_feature": fused_feature,
        "pseudo_logits_low": pseudo_logits_low,
        "block11_feature": block11_feature,
    }


def save_tensor_bundle(
    tensor: torch.Tensor,
    output_dir: str,
    stem: str,
    summary_mode: str,
    spatial_cmap: str,
):
    ensure_dir(output_dir)
    spatial_map = feature_spatial_summary(tensor, summary_mode)
    spectrum_map = mean_log_power_spectrum(tensor)

    np.save(os.path.join(output_dir, f"{stem}_raw.npy"), tensor.detach().cpu().numpy())
    np.save(os.path.join(output_dir, f"{stem}_summary.npy"), spatial_map.astype(np.float32))
    np.save(os.path.join(output_dir, f"{stem}_spectrum.npy"), spectrum_map.astype(np.float32))

    save_heatmap(
        normalize_map(spatial_map),
        os.path.join(output_dir, f"{stem}_spatial.png"),
        cmap=spatial_cmap,
        title=f"{stem} spatial summary",
    )
    save_heatmap(
        normalize_map(spectrum_map),
        os.path.join(output_dir, f"{stem}_spectrum.png"),
        cmap="inferno",
        title=f"{stem} mean log power spectrum",
    )
    save_panel(
        spatial_map=spatial_map,
        spectrum_map=spectrum_map,
        path=os.path.join(output_dir, f"{stem}_panel.png"),
        title=stem,
        spatial_cmap=spatial_cmap,
    )


def main():
    args = parse_args()

    ensure_dir(args.output_dir)
    dataset = build_dataset(args)
    sample_index = find_sample_index(dataset, args.sample_name) if args.sample_name else args.sample_index

    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(f"sample_index {sample_index} out of range 0-{len(dataset) - 1}")

    sample_name, image, label, cls_label = dataset[sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.from_numpy(np.asarray(image)).unsqueeze(0).float().to(device)
    label = torch.from_numpy(np.asarray(label)).unsqueeze(0).long().to(device)
    cls_label = torch.from_numpy(np.asarray(cls_label)).unsqueeze(0).float().to(device)

    model = DiffusionBasedNetwork(
        num_classes=args.num_classes,
        backbone=args.diffusion_model,
        attention_layers_to_use=args.attention_layers_to_use,
        dataset_name="voc2012",
        feature_extractor=args.feature_extractor,
    ).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    model.fuser.extractor.scheduler.set_timesteps(1000)

    outputs = extract_requested_tensors(model, image, cls_label)
    active_classes = outputs["active_classes"].tolist()
    active_class_names = [voc.class_list[idx + 1] for idx in active_classes]

    sample_dir = os.path.join(args.output_dir, sample_name)
    ensure_dir(sample_dir)
    imageio.imwrite(os.path.join(sample_dir, "input_image.png"), denormalize_rgb(image))

    metadata = {
        "sample_name": sample_name,
        "sample_index": int(sample_index),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "stage": get_stage(args.split, args.stage),
        "active_classes": active_classes,
        "active_class_names": active_class_names,
        "label_shape": list(label.shape[-2:]),
        "feature_shapes": [list(feat.shape) for feat in outputs["all_features"]],
        "block11_shape": list(outputs["block11_feature"].shape),
        "note": "all_features follows the current model/extractor.py implementation and corresponds to UNet up-block features at t=0. With the default block_indices, the last entry is block11 at 64x64.",
        "summary_modes": {
            "att_map": "mean over active class channels",
            "fused_feature": "L2 norm over channels",
            "pseudo_logits_low": "max softmax confidence over classes",
            "block11_feature": "L2 norm over channels",
        },
    }
    with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    att_dir = os.path.join(sample_dir, "att_map")
    fused_dir = os.path.join(sample_dir, "fused_feature")
    logits_dir = os.path.join(sample_dir, "pseudo_logits_low")
    block11_dir = os.path.join(sample_dir, "block11")
    ensure_dir(att_dir)
    ensure_dir(fused_dir)
    ensure_dir(logits_dir)
    ensure_dir(block11_dir)

    save_tensor_bundle(
        tensor=outputs["att_map"],
        output_dir=att_dir,
        stem="att_map",
        summary_mode="attention_mean",
        spatial_cmap="jet",
    )
    save_channel_grid(
        tensor=outputs["att_map"],
        path=os.path.join(att_dir, "att_map_channels.png"),
        title="att_map per active class",
        channel_names=active_class_names,
        cmap="jet",
    )

    save_tensor_bundle(
        tensor=outputs["fused_feature"],
        output_dir=fused_dir,
        stem="fused_feature",
        summary_mode="feature_l2",
        spatial_cmap="magma",
    )

    save_tensor_bundle(
        tensor=outputs["pseudo_logits_low"],
        output_dir=logits_dir,
        stem="pseudo_logits_low",
        summary_mode="logit_confidence",
        spatial_cmap="viridis",
    )
    logit_channel_ids = [0] + [idx + 1 for idx in active_classes]
    logit_channel_names = [voc.class_list[idx] for idx in logit_channel_ids]
    save_channel_grid(
        tensor=outputs["pseudo_logits_low"][0, logit_channel_ids],
        path=os.path.join(logits_dir, "pseudo_logits_low_channels.png"),
        title="pseudo_logits_low selected channels",
        channel_names=logit_channel_names,
        cmap="viridis",
    )

    save_tensor_bundle(
        tensor=outputs["block11_feature"],
        output_dir=block11_dir,
        stem="block11",
        summary_mode="feature_l2",
        spatial_cmap="magma",
    )

    print(f"[INFO] Saved requested visualizations to {sample_dir}")


if __name__ == "__main__":
    main()
