import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.cd8_training import COLOR_MAP, resolve_device
from .dataset import INPUT_MASK_NAMES, OnchipCD8Dataset, load_split_json
from common.cd8_model import UNet

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run inference for on-chip CD8 segmentation.")
    parser.add_argument("--model-path", type=Path, default=Path("onchip_cd8_distribution/results/seed-1_lr-0p0001_bs-8_wd-0p0001_base-32_input-dapi_actin_ck/best_model.pth"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data/data_split.json"))
    parser.add_argument("--split-set", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--mask-names", nargs="+", default=list(INPUT_MASK_NAMES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    r, g, b = color
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (mask * float(r)).astype(np.uint8)
    rgb[..., 1] = (mask * float(g)).astype(np.uint8)
    rgb[..., 2] = (mask * float(b)).astype(np.uint8)
    return rgb


def make_combined_rgb(input_masks: dict[str, np.ndarray], mask_names: list[str]) -> np.ndarray:
    first = next(iter(input_masks.values()))
    h, w = first.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    if "tissue" in input_masks:
        alpha = input_masks["tissue"][..., np.newaxis]
        tissue_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        canvas = canvas * (1.0 - alpha) + tissue_color * alpha
    for name in mask_names:
        if name == "tissue" or name not in input_masks:
            continue
        color = np.array(COLOR_MAP.get(str(name).lower(), (255, 255, 255)), dtype=np.float32) / 255.0
        alpha = input_masks[name][..., np.newaxis]
        canvas = canvas * (1.0 - alpha) + color * alpha
    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def _legend_order(mask_names: list[str]) -> list[str]:
    names = list(mask_names)
    ordered = []
    if "tissue" in names:
        ordered.append("tissue")
    for name in names:
        if name != "tissue":
            ordered.append(name)
    return ordered


def _display_name(name: str) -> str:
    return str(name).replace("_", " ").upper()


def save_combined_with_legend(combined_rgb: np.ndarray, mask_names: list[str], out_path: Path) -> None:
    h, w = int(combined_rgb.shape[0]), int(combined_rgb.shape[1])
    ordered_names = _legend_order(mask_names)
    legend_w = 280
    item_h = 28
    top_pad = 20
    bottom_pad = 20
    legend_h = top_pad + bottom_pad + max(1, len(ordered_names)) * item_h
    canvas_h = max(h, legend_h)
    canvas = Image.new("RGB", (w + legend_w, canvas_h), color=(255, 255, 255))
    combined_img = Image.fromarray(combined_rgb, mode="RGB")
    canvas.paste(combined_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x0 = w + 16
    y = top_pad
    for name in ordered_names:
        color = tuple(COLOR_MAP.get(str(name).lower(), (255, 255, 255)))
        draw.rectangle([x0, y + 4, x0 + 18, y + 22], fill=color, outline=(0, 0, 0), width=1)
        draw.text((x0 + 28, y + 5), _display_name(name), fill=(0, 0, 0), font=font)
        y += item_h
    canvas.save(out_path)


@torch.no_grad()
def run_inference(
    model_path: Path,
    masks_dir: Path,
    split_json: Path,
    split_set: str,
    output_dir: Path,
    image_size: int,
    base_channels: int,
    mask_names: list,
    batch_size: int,
    num_workers: int,
    device_str: str,
) -> None:
    device = resolve_device(device_str)
    state_dict = torch.load(model_path, map_location=device)
    model_in_channels = int(state_dict.get("enc1.block.0.weight", torch.empty(1, len(mask_names))).shape[1])
    split_payload = load_split_json(split_json)
    case_ids = split_payload.get(split_set, []) or []
    if not case_ids:
        raise ValueError(f"No case IDs found for split '{split_set}' in {split_json}")

    dataset, actual_mask_names = _build_dataset_with_mask_fallback(
        masks_dir=masks_dir,
        case_ids=list(case_ids),
        image_size=image_size,
        model_in_channels=model_in_channels,
        requested_mask_names=list(mask_names),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=model_in_channels, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_case_counts = {}

    for x_norm, y, case_ids_batch in tqdm(loader, desc=f"Inference ({split_set})"):
        x_norm = x_norm.to(device)
        y = y.to(device)
        probs = torch.sigmoid(model(x_norm))
        x_raw = (x_norm * 0.5 + 0.5).clamp(0.0, 1.0)
        x_np = x_raw.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        for i, case_id in enumerate(case_ids_batch):
            k = seen_case_counts.get(case_id, 0)
            seen_case_counts[case_id] = k + 1
            folder_name = case_id if k == 0 else f"{case_id}_{k}"
            sample_dir = output_dir / folder_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            input_masks_dict: dict[str, np.ndarray] = {}
            for ch_idx, name in enumerate(actual_mask_names):
                mask_2d = np.clip(x_np[i, ch_idx], 0.0, 1.0)
                input_masks_dict[str(name)] = mask_2d
                color = COLOR_MAP.get(str(name).lower(), (255, 255, 255))
                Image.fromarray(mask_to_rgb(mask_2d, color), mode="RGB").save(sample_dir / f"{name}_mask.png")
            combined = make_combined_rgb(input_masks_dict, list(actual_mask_names))
            save_combined_with_legend(combined, list(actual_mask_names), sample_dir / "combined_input_mask.png")
            label_2d = np.clip(labels_np[i, 0], 0.0, 1.0)
            pred_2d = np.clip(probs_np[i, 0], 0.0, 1.0)
            cd8_color = COLOR_MAP.get("cd8", (255, 77, 37))
            Image.fromarray(mask_to_rgb(label_2d, cd8_color), mode="RGB").save(sample_dir / "label_cd8.png")
            Image.fromarray(mask_to_rgb(pred_2d, cd8_color), mode="RGB").save(sample_dir / "pred_cd8.png")


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "")


def _pick_mask_names(model_in_channels: int, requested_mask_names: list[str]) -> list[str]:
    if len(requested_mask_names) == model_in_channels:
        return list(requested_mask_names)
    presets = {
        3: ["dapi", "actin", "ck"],
        2: ["actin", "ck"],
        1: ["actin"],
    }
    if model_in_channels in presets:
        return list(presets[model_in_channels])
    if model_in_channels == len(INPUT_MASK_NAMES):
        return list(INPUT_MASK_NAMES)
    return [f"ch{i}" for i in range(model_in_channels)]


def _candidate_mask_name_sets(model_in_channels: int, requested_mask_names: list[str]) -> list[list[str]]:
    candidates = []
    primary = _pick_mask_names(model_in_channels, requested_mask_names)
    candidates.append(list(primary))
    presets = {
        3: ["dapi", "actin", "ck"],
        2: ["actin", "ck"],
        1: ["actin"],
    }
    if model_in_channels in presets:
        candidates.append(list(presets[model_in_channels]))
    if model_in_channels == len(INPUT_MASK_NAMES):
        candidates.append(list(INPUT_MASK_NAMES))
    unique = []
    seen = set()
    for names in candidates:
        key = tuple(_normalize_name(x) for x in names)
        if key in seen:
            continue
        seen.add(key)
        unique.append(names)
    return unique


def _build_dataset_with_mask_fallback(
    masks_dir: Path,
    case_ids: list[str],
    image_size: int,
    model_in_channels: int,
    requested_mask_names: list[str],
) -> tuple[OnchipCD8Dataset, list[str]]:
    last_err = None
    for names in _candidate_mask_name_sets(model_in_channels, requested_mask_names):
        try:
            ds = OnchipCD8Dataset(
                masks_dir=masks_dir,
                case_ids=list(case_ids),
                input_mask_names=list(names),
                target_mask_name="cd8",
                image_size=image_size,
                augment=False,
                normalize=True,
            )
            return ds, list(names)
        except ValueError as e:
            last_err = e
    if last_err is not None:
        raise ValueError(
            f"{last_err} model_in_channels={model_in_channels}, requested_mask_names={list(requested_mask_names)}, "
            f"tried={_candidate_mask_name_sets(model_in_channels, requested_mask_names)}"
        ) from last_err
    raise RuntimeError("Failed to build dataset.")


def main(argv=None) -> int:
    args = parse_args(argv)
    mask_names = list(args.mask_names)
    if len(mask_names) == 1 and "," in mask_names[0]:
        mask_names = [s.strip() for s in mask_names[0].split(",") if s.strip()]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.model_path.parent / f"{args.split_set}_predictions"
    run_inference(
        model_path=args.model_path,
        masks_dir=args.masks_dir,
        split_json=args.split_json,
        split_set=args.split_set,
        output_dir=output_dir,
        image_size=args.image_size,
        base_channels=args.base_channels,
        mask_names=mask_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
