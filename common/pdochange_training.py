import numpy as np
import torch


def resolve_device(device_str: str = "cuda") -> torch.device:
    d = (device_str or "cuda").strip().lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _accumulate_classification(
    logits_list: list[torch.Tensor], targets: list[torch.Tensor]
) -> tuple[float, float]:
    if not logits_list or not targets:
        return float("nan"), float("nan")
    logits_all = torch.cat(logits_list, dim=0)
    target_all = torch.cat(targets, dim=0).long()
    pred_all = torch.argmax(logits_all, dim=1)
    acc = float((pred_all == target_all).float().mean().item())
    num_classes = int(logits_all.shape[1])
    f1_scores = []
    for c in range(num_classes):
        pred_c = pred_all == c
        true_c = target_all == c
        tp = int((pred_c & true_c).sum().item())
        fp = int((pred_c & (~true_c)).sum().item())
        fn = int(((~pred_c) & true_c).sum().item())
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall <= 1e-12:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_scores.append(float(f1))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else float("nan")
    return acc, macro_f1


def epoch_train(model, loader, optimizer, criterion, device: torch.device):
    model.train()
    loss_sum = 0.0
    count = 0
    logits_list: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y, _ids in loader:
        x = x.to(device)
        y = y.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bsz = int(x.shape[0])
        loss_sum += float(loss.item()) * bsz
        count += bsz
        logits_list.append(logits.detach().cpu())
        targets.append(y.detach().cpu())
    avg_loss = loss_sum / max(count, 1)
    acc, macro_f1 = _accumulate_classification(logits_list, targets)
    return avg_loss, acc, macro_f1


@torch.no_grad()
def epoch_eval(model, loader, criterion, device: torch.device):
    model.eval()
    loss_sum = 0.0
    count = 0
    logits_list: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    ids: list[str] = []
    for x, y, batch_ids in loader:
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = criterion(logits, y)
        bsz = int(x.shape[0])
        loss_sum += float(loss.item()) * bsz
        count += bsz
        logits_list.append(logits.detach().cpu())
        targets.append(y.detach().cpu())
        ids.extend([str(v) for v in batch_ids])
    avg_loss = loss_sum / max(count, 1)
    acc, macro_f1 = _accumulate_classification(logits_list, targets)
    if logits_list:
        logits_all = torch.cat(logits_list, dim=0)
        prob_all = torch.softmax(logits_all, dim=1)
        pred_idx = torch.argmax(prob_all, dim=1).numpy()
        pred_conf = torch.max(prob_all, dim=1).values.numpy()
    else:
        pred_idx = np.empty((0,), dtype=np.int64)
        pred_conf = np.empty((0,), dtype=np.float32)
    target_all = torch.cat(targets, dim=0).numpy() if targets else np.empty((0,), dtype=np.int64)
    return avg_loss, acc, macro_f1, ids, pred_idx, target_all, pred_conf
