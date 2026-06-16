from pathlib import Path

import numpy as np
import torch
from PIL import Image

COLOR_MAP = {
    "cd8": (255, 77, 37),
    "cd4": (22, 80, 170),
    "cd68": (240, 230, 96),
    "ck": (231, 219, 145),
    "actin": (153, 214, 224),
    "dapi": (90, 110, 255),
    "pd-1": (119, 32, 29),
    "tissue": (255, 255, 255),
}


def _mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    r, g, b = color
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (mask * float(r)).astype(np.uint8)
    rgb[..., 1] = (mask * float(g)).astype(np.uint8)
    rgb[..., 2] = (mask * float(b)).astype(np.uint8)
    return rgb


def resolve_device(device_str: str) -> torch.device:
    if str(device_str).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if str(device_str).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def rmse_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return torch.sqrt(torch.mean((probs - targets) ** 2))


def epoch_train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_rmse = 0.0
    num_batches = 0
    for x, y, _case_ids in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dice = dice_score_from_logits(logits, y)
            rmse = rmse_from_logits(logits, y)
        total_loss += float(loss.item())
        total_dice += float(dice.item())
        total_rmse += float(rmse.item())
        num_batches += 1
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches, total_rmse / num_batches


@torch.no_grad()
def epoch_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_rmse = 0.0
    num_batches = 0
    for x, y, _case_ids in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        dice = dice_score_from_logits(logits, y)
        rmse = rmse_from_logits(logits, y)
        total_loss += float(loss.item())
        total_dice += float(dice.item())
        total_rmse += float(rmse.item())
        num_batches += 1
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / num_batches, total_dice / num_batches, total_rmse / num_batches


@torch.no_grad()
def evaluate_and_save_val_predictions(
    model,
    loader,
    criterion,
    device,
    output_dir: Path,
    input_mask_names: list[str] | None = None,
) -> dict:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_loss = 0.0
    total_dice = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    num_batches = 0
    seen_case_counts = {}
    mask_names = list(input_mask_names) if input_mask_names is not None else []
    for x, y, case_ids in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        loss = criterion(logits, y)
        dice = dice_score_from_logits(logits, y)
        rmse = rmse_from_logits(logits, y)
        mae = torch.mean(torch.abs(probs - y))
        total_loss += float(loss.item())
        total_dice += float(dice.item())
        total_rmse += float(rmse.item())
        total_mae += float(mae.item())
        num_batches += 1
        x_raw_np = (x.detach().cpu().numpy() * 0.5 + 0.5).clip(0.0, 1.0)
        probs_np = probs.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        for i, case_id in enumerate(case_ids):
            k = seen_case_counts.get(case_id, 0)
            seen_case_counts[case_id] = k + 1
            suffix = "" if k == 0 else f"_{k}"
            pred_img = (np.clip(probs_np[i, 0], 0.0, 1.0) * 255.0).astype(np.uint8)
            label_img = (np.clip(labels_np[i, 0], 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(pred_img).save(output_dir / f"{case_id}{suffix}_pred_cd8.png")
            Image.fromarray(label_img).save(output_dir / f"{case_id}{suffix}_label_cd8.png")
            for ch_idx, name in enumerate(mask_names):
                if ch_idx >= int(x_raw_np.shape[1]):
                    break
                color = COLOR_MAP.get(str(name).lower(), (255, 255, 255))
                rgb_img = _mask_to_rgb(np.clip(x_raw_np[i, ch_idx], 0.0, 1.0), color)
                safe_name = str(name).replace("/", "_").replace(" ", "_")
                Image.fromarray(rgb_img).save(
                    output_dir / f"{case_id}{suffix}_input_{safe_name}.png"
                )
    if num_batches == 0:
        return {"val_loss": 0.0, "val_dice": 0.0, "val_rmse": 0.0, "val_mae": 0.0}
    return {
        "val_loss": total_loss / num_batches,
        "val_dice": total_dice / num_batches,
        "val_rmse": total_rmse / num_batches,
        "val_mae": total_mae / num_batches,
    }
