import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import math
import multiprocessing as mp
import os
import shutil
import traceback
import openslide
from tqdm import tqdm
from PIL import Image
from numpy.lib.format import open_memmap

from preprocess.tcga_wsi.gigatime import GigaTIME


# ============================
# 1. Dataset: 负责切片读取和预处理
# ============================
class WSIGridBatchDataset(Dataset):
    """
    高性能 WSI 读取器：一次读取一大块 (Super Tile)，然后在内存中切分成小 Patch。
    这样大大减少了 OpenSlide 的 read_region 调用次数。
    """
    def __init__(self, slide_path, target_mpp, patch_size, stride, batch_size):
        self.slide_path = slide_path
        self.target_mpp = target_mpp
        self.patch_size = patch_size
        self.stride = stride
        self.target_batch_size = batch_size
        
        # 1. 计算 "Super Tile" (大块) 的大小
        # 我们希望一次读取尽可能凑够 batch_size 个 patch
        # 为了简单，假设我们按行读取或者按矩形块读取。
        # 这里采用：根据 batch_size 估算一个 NxN 的大块。
        # 例如 batch_size=64, 我们尝试读一个 8x8 grid 的大区域
        self.grid_side = int(np.ceil(np.sqrt(batch_size))) # 比如 sqrt(32) -> 6
        
        # 实际一次读取的物理尺寸 (Target分辨率下)
        self.super_tile_w = self.grid_side * stride + (patch_size - stride)
        self.super_tile_h = self.grid_side * stride + (patch_size - stride)

        # 2. 初始化 Slide 信息
        _slide = openslide.OpenSlide(slide_path)
        try:
            mpp_x = float(_slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
            mpp_y = float(_slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
            self.native_mpp = (mpp_x + mpp_y) / 2.0
        except:
            self.native_mpp = 0.25
        
        self.downsample_factor = self.target_mpp / self.native_mpp
        w_0, h_0 = _slide.dimensions
        self.target_w = int(w_0 / self.downsample_factor)
        self.target_h = int(h_0 / self.downsample_factor)
        _slide.close()

        # 3. 生成 "Super Tile" 的坐标列表
        # 这次我们的 coords 存的是大块的左上角坐标
        self.super_coords = []
        # 注意：这里的步长是 grid_side * stride
        step_size = self.grid_side * stride
        
        for y in range(0, self.target_h, step_size):
            for x in range(0, self.target_w, step_size):
                self.super_coords.append((x, y))

    def _get_slide(self):
        if not hasattr(self, '_worker_slide'):
            self._worker_slide = openslide.OpenSlide(self.slide_path)
        return self._worker_slide

    def __len__(self):
        return len(self.super_coords)

    def __getitem__(self, idx):
        # 1. 获取大块坐标
        sx, sy = self.super_coords[idx]
        slide = self._get_slide()
        
        # 2. 读取大块 (IO 操作，只做一次)
        l0_x = int(sx * self.downsample_factor)
        l0_y = int(sy * self.downsample_factor)
        
        # 计算需要读取的 Level0 尺寸
        read_w = int(self.super_tile_w * self.downsample_factor)
        read_h = int(self.super_tile_h * self.downsample_factor)
        
        # 避免读出边界
        # (这里为了高性能，openslide允许越界读取，会自动填充0或透明，也可以手动处理)
        try:
            big_img = slide.read_region((l0_x, l0_y), 0, (read_w, read_h))
            big_img = big_img.convert("RGB")
            
            # Resize 到目标 Target 分辨率
            if read_w != self.super_tile_w or read_h != self.super_tile_h:
                big_img = big_img.resize((self.super_tile_w, self.super_tile_h), Image.BILINEAR)
            
            big_arr = np.array(big_img) # (H_big, W_big, 3)
            
        except Exception as e:
            # 容错
            big_arr = np.zeros((self.super_tile_h, self.super_tile_w, 3), dtype=np.uint8)

        # 3. 内存切片 (CPU 计算，极快)
        batch_imgs = []
        batch_coords = []
        
        for i in range(self.grid_side):     # y 方向
            for j in range(self.grid_side): # x 方向
                # 相对偏移
                rel_y = i * self.stride
                rel_x = j * self.stride
                
                # 绝对坐标
                curr_x = sx + rel_x
                curr_y = sy + rel_y
                
                # 检查是否超出图像边界
                if curr_x >= self.target_w or curr_y >= self.target_h:
                    continue
                    
                # 切出 Patch
                # 注意：我们的大图可能因为边界问题比预期的 grid 小，需要防越界
                patch = big_arr[rel_y : rel_y + self.patch_size, 
                                rel_x : rel_x + self.patch_size, :]
                                
                # 如果 patch 不完整 (边缘)，进行 Padding
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    pad_h = self.patch_size - patch.shape[0]
                    pad_w = self.patch_size - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

                batch_imgs.append(patch)
                batch_coords.append([curr_x, curr_y])

        # 4. 统一预处理 (Vectorized/Batch processing)
        if len(batch_imgs) == 0:
             # 极其边缘的情况，返回空
             return torch.empty(0), torch.empty(0)

        batch_imgs = np.stack(batch_imgs) # (B, 256, 256, 3)
        batch_imgs = batch_imgs.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        batch_imgs = (batch_imgs - mean) / std
        
        batch_imgs = np.transpose(batch_imgs, (0, 3, 1, 2)) # (B, C, H, W)
        
        # 返回整个 Batch
        return torch.from_numpy(batch_imgs.astype(np.float32)), torch.tensor(batch_coords)

# ============================
# 2. Stitcher: 负责把结果拼回去
# ============================
class NpyStitcher:
    def __init__(
        self,
        *,
        output_path: Path,
        cache_dir: Path,
        shape: tuple[int, int],
        channels: int,
        tile_size: int = 2048,
        resume: bool = False,
    ):
        self.output_path = Path(output_path)
        self.cache_dir = Path(cache_dir)
        self.shape = (int(shape[0]), int(shape[1]))  # (H, W)
        self.channels = int(channels)
        self.packed_channels = int((self.channels + 1) // 2)
        self.tile_size = int(tile_size)

        self.sum_path = self.cache_dir / "sum.npy"
        self.count_path = self.cache_dir / "count.npy"

        if resume and self.sum_path.exists() and self.count_path.exists():
            self.sum_mm = open_memmap(str(self.sum_path), mode="r+")
            self.count_mm = open_memmap(str(self.count_path), mode="r+")
        else:
            if self.sum_path.exists():
                self.sum_path.unlink()
            if self.count_path.exists():
                self.count_path.unlink()
            self.sum_mm = open_memmap(
                str(self.sum_path),
                mode="w+",
                dtype=np.float32,
                shape=(self.shape[0], self.shape[1], self.channels),
            )
            self.count_mm = open_memmap(
                str(self.count_path),
                mode="w+",
                dtype=np.uint16,
                shape=(self.shape[0], self.shape[1]),
            )
            self.sum_mm[:] = 0
            self.count_mm[:] = 0

    def add_batch(self, coords: np.ndarray, predictions: np.ndarray):
        predictions = np.transpose(predictions, (0, 2, 3, 1))  # (B, H, W, C)
        b, h, w, c = predictions.shape
        for i in range(b):
            x, y = int(coords[i, 0]), int(coords[i, 1])
            pred = predictions[i]

            h_end = min(y + h, self.shape[0])
            w_end = min(x + w, self.shape[1])
            h_len = h_end - y
            w_len = w_end - x
            if h_len <= 0 or w_len <= 0:
                continue
            pred_crop = pred[:h_len, :w_len, :]

            self.sum_mm[y:h_end, x:w_end, :] += pred_crop
            self.count_mm[y:h_end, x:w_end] += 1

    def finalize(self):
        if self.output_path.exists():
            self.output_path.unlink()

        out_mm = open_memmap(
            str(self.output_path),
            mode="w+",
            dtype=np.uint8,
            shape=(self.shape[0], self.shape[1], self.packed_channels),
        )

        h, w = self.shape
        ts = self.tile_size
        for y0 in tqdm(range(0, h, ts), desc="Finalizing"):
            y1 = min(y0 + ts, h)
            for x0 in range(0, w, ts):
                x1 = min(x0 + ts, w)
                s = self.sum_mm[y0:y1, x0:x1, :]
                c = self.count_mm[y0:y1, x0:x1].astype(np.float32)
                c = np.maximum(c, 1.0)
                mean = s / c[:, :, None]
                pred = (mean > 0.5).astype(np.uint8, copy=False)

                even = pred[:, :, 0::2] & 0x0F
                packed = np.zeros((pred.shape[0], pred.shape[1], self.packed_channels), dtype=np.uint8)
                packed[:, :, :even.shape[2]] = even
                odd = pred[:, :, 1::2] & 0x0F
                packed[:, :, :odd.shape[2]] |= (odd << 4)
                out_mm[y0:y1, x0:x1, :] = packed

        out_mm.flush()
        try:
            del out_mm
        except Exception:
            pass

        try:
            self.sum_mm.flush()
            self.count_mm.flush()
        except Exception:
            pass

    def flush(self):
        self.sum_mm.flush()
        self.count_mm.flush()


# ============================
# 3. Main Logic
# ============================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/TCGA_WSI_part4")
    parser.add_argument("--output-dir", default="data/TCGA_WSI_part4_output")
    parser.add_argument("--model-weights", default="data/hf_models/prov-gigatime--GigaTIME/model.pth")
    parser.add_argument("--batch-size", type=int, default=49) # 可以设置大了！
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--channels", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-downsample", type=int, default=16)
    parser.add_argument("--output-dtype", choices=["int4"], default="int4")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--resume-from-batch", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. 准备模型
    device = torch.device(args.device)
    print(f"Loading model from {args.model_weights}...")
    
    # 初始化模型结构
    model = GigaTIME(num_classes=args.channels, input_channels=3, deep_supervision=False)
    
    # 加载权重
    state_dict = torch.load(args.model_weights, map_location="cpu")
    # 处理可能的 DataParallel module 前缀
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k[7:]] = v
        state_dict = new_state
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. 遍历文件
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    svs_files = list(input_dir.glob("*.svs"))
    if not svs_files:
        print("No .svs files found.")
        return

    for idx, svs_path in enumerate(svs_files):
        print(f"\n[{idx+1}/{len(svs_files)}] Processing: {svs_path.name}")

        output_npy = output_dir / f"{svs_path.stem}.npy"
        if output_npy.exists() and not args.overwrite:
            print("  Skip: output already exists.")
            continue

        cache_dir = output_dir / f"{svs_path.stem}_cache"
        progress_path = cache_dir / "progress.json"
        if args.overwrite and cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 创建 Dataset 和 Loader
        # 参数: 目标MPP=0.5, Patch=256, Stride=128 (50% Overlap)
        dataset = WSIGridBatchDataset(
            str(svs_path), 
            target_mpp=args.mpp, 
            patch_size=args.patch_size, 
            stride=args.stride,
            batch_size=args.batch_size 
        )

        start_batch = 0
        if args.resume_from_batch is not None:
            start_batch = int(args.resume_from_batch)
        elif not args.no_resume and progress_path.exists():
            try:
                start_batch = int(json.loads(progress_path.read_text()).get("next_batch", 0))
            except Exception:
                start_batch = 0
        start_batch = max(0, min(start_batch, len(dataset)))

        dataset_for_loader = dataset if start_batch == 0 else Subset(dataset, range(start_batch, len(dataset)))

        loader = DataLoader(
            dataset_for_loader,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.num_workers,
        )
        
        # 4. 初始化 Stitcher
        ds = int(args.output_downsample)
        if ds <= 0:
            raise ValueError("--output-downsample must be >= 1")

        out_h = int(math.ceil(dataset.target_h / float(ds)))
        out_w = int(math.ceil(dataset.target_w / float(ds)))
        
        print(f"  Target Shape: {dataset.target_w} x {dataset.target_h}")
        print(f"  Output Shape: {out_w} x {out_h} (downsample={ds})")
        print(f"  Total Patches: {len(dataset)}")

        stitcher = NpyStitcher(
            output_path=output_npy,
            cache_dir=cache_dir,
            shape=(out_h, out_w),
            channels=args.channels,
            resume=(start_batch > 0 and not args.overwrite),
        )

        with torch.no_grad():
            for local_idx, (imgs, coords) in enumerate(tqdm(loader, desc="Inference")):
                batch_idx = start_batch + int(local_idx)
                imgs = imgs.squeeze(0)
                coords = coords.squeeze(0)

                if imgs.shape[0] == 0:
                    tmp_path = cache_dir / "progress.json.tmp"
                    tmp_path.write_text(json.dumps({"next_batch": batch_idx + 1}))
                    os.replace(tmp_path, progress_path)
                    continue

                cache_path = cache_dir / f"batch_{batch_idx:06d}.npz"
                if cache_path.exists():
                    cached = np.load(cache_path)
                    coords_np = cached["coords"]
                    preds_u8 = cached["preds"]
                else:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                    probs = torch.sigmoid(logits)  # (B, C, H, W)
                    probs_np = probs.detach().cpu().numpy()
                    coords_np = coords.numpy().astype(np.int32)

                    if ds > 1:
                        probs_np = probs_np[:, :, ::ds, ::ds]
                        coords_np = (coords_np // ds).astype(np.int32)

                    preds_u8 = (probs_np > 0.5).astype(np.uint8, copy=False)
                    np.savez(cache_path, coords=coords_np, preds=preds_u8)

                stitcher.add_batch(coords_np, preds_u8.astype(np.float32, copy=False))
                stitcher.flush()
                tmp_path = cache_dir / "progress.json.tmp"
                tmp_path.write_text(json.dumps({"next_batch": batch_idx + 1}))
                os.replace(tmp_path, progress_path)

        stitcher.finalize()

        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"  Saved: {output_npy}")
        print("  Done.")


if __name__ == "__main__":
    main()
