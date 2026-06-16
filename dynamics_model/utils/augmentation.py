import numpy as np
import torch
import torch.nn.functional as F


def mixup_generate(x_seq, x_track, y, x_pdosize=None, x_antigen=None, n=10000, alpha=0.2, device="cpu"):
    x_seq = x_seq.to(device)
    x_track = x_track.to(device)
    y = torch.tensor(y).to(device)
    if x_pdosize is not None:
        x_pdosize = x_pdosize.to(device)
    if x_antigen is not None:
        x_antigen = x_antigen.to(device)
    N = x_seq.size(0)
    idx1 = torch.randint(0, N, (n,), device=device)
    idx2 = torch.randint(0, N, (n,), device=device)
    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=n).astype(np.float32)).to(device)
    lam_seq = lam.view(-1, 1, 1)
    lam_vec = lam.view(-1, 1)
    x_seq_mix = x_seq[idx1] + lam_seq * x_seq[idx2]
    x_track_mix = x_track[idx1] + lam_vec * x_track[idx2]
    x_pdo_mix = None
    x_antigen_mix = None
    if x_pdosize is not None:
        x_pdo_mix = x_pdosize[idx1] + lam_vec * x_pdosize[idx2]
    if x_antigen is not None:
        x_antigen_mix = x_antigen[idx1] + lam_vec * x_antigen[idx2]
    num_classes = int(torch.max(y).item()) + 1
    y1 = F.one_hot(y[idx1], num_classes=num_classes).float()
    y2 = F.one_hot(y[idx2], num_classes=num_classes).float()
    y_mix = y1 + lam_vec * y2
    return {
        "x_seq": x_seq_mix,
        "x_track": x_track_mix,
        "x_pdosize": x_pdo_mix,
        "x_antigen": x_antigen_mix,
        "y_soft": y_mix,
    }
