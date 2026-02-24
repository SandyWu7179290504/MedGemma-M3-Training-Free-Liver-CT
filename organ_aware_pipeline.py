#!/usr/bin/env python3
"""
Organ-Aware MedGemma 1.5 Pipeline & Ablation Study
====================================================
使用 TotalSegmentator 官方測試資料 (example_ct_sm.nii.gz + example_seg.nii.gz)
進行器官感知裁剪，並與 MedGemma 1.5 整合進行 Ablation 比較。

實作內容:
  1. 載入 TotalSegmentator 官方測試 CT + 預算分割 Mask
  2. 依據真實分割 Mask 擷取器官 ROI
  3. MedGemma 1.5 推論: Full Image vs. Organ-Aware Cropped
  4. Ablation 比較: IoU 指標 + 並排視覺化
  5. 座標映射: 裁剪區域座標 → 原始影像座標

使用方式:
  # 使用官方測試資料 (預設)
  python organ_aware_pipeline.py

  # 指定器官
  python organ_aware_pipeline.py --organ liver

  # 使用自訂 CT + TotalSegmentator 即時分割
  python organ_aware_pipeline.py --input /path/to/ct.nii.gz --organ kidney_right

  # 使用 2D 影像
  python organ_aware_pipeline.py --input /path/to/image.png --organ lung
"""

import os
import sys
import json
import re
import gc
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 無頭模式
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 常數定義
# ============================================================
MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
MEDGEMMA_INPUT_SIZE = 896          # MedGemma 需要 896x896
COORD_NORM = 1000                  # MedGemma 0-1000 正規化座標
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "ablation_results"
TEST_DATA_DIR = SCRIPT_DIR / "test_data"

# HU-to-RGB 視窗設定 (依照 MedGemma 規範)
HU_WINDOWS = {
    "R": (-1024, 1024),   # 中心 0,   寬度 2048
    "G": (-135,  215),    # 中心 40,  寬度 350  (軟組織)
    "B": (0,     80),     # 中心 40,  寬度 80   (腦部/細緻)
}

# TotalSegmentator 官方 Label ID → 器官名稱 (依照 GitHub class_map)
# https://github.com/wasserth/TotalSegmentator#class-details
ORGAN_LABEL_MAP = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
    "lung_upper_lobe_left": 10,
    "lung_lower_lobe_left": 11,
    "lung_upper_lobe_right": 12,
    "lung_middle_lobe_right": 13,
    "lung_lower_lobe_right": 14,
    "esophagus": 15,
    "trachea": 16,
    "thyroid_gland": 17,
    "small_bowel": 18,
    "duodenum": 19,
    "colon": 20,
    "urinary_bladder": 21,
    "prostate": 22,
    "sacrum": 25,
    "heart": 51,
    "aorta": 52,
    "pulmonary_vein": 53,
    "superior_vena_cava": 62,
    "inferior_vena_cava": 63,
    "brain": 90,
    "skull": 91,
}

# 反向映射: label_id → organ_name
LABEL_ID_TO_NAME = {v: k for k, v in ORGAN_LABEL_MAP.items()}


# ============================================================
# 1. 工具函式
# ============================================================

def hu_to_rgb(hu_slice: np.ndarray) -> np.ndarray:
    """
    將 HU 值的 CT 切片轉換為 3 通道 RGB 影像。
    R 通道: HU 視窗 (-1024, 1024)  — 全範圍
    G 通道: HU 視窗 (-135,  215)   — 軟組織
    B 通道: HU 視窗 (0,     80)    — 細緻對比
    """
    rgb = np.zeros((*hu_slice.shape, 3), dtype=np.uint8)
    for ch_idx, (ch_name, (low, high)) in enumerate(HU_WINDOWS.items()):
        clipped = np.clip(hu_slice, low, high)
        normalized = ((clipped - low) / (high - low) * 255).astype(np.uint8)
        rgb[:, :, ch_idx] = normalized
    return rgb


def pad_to_square(image_array: np.ndarray) -> np.ndarray:
    """將影像 padding 成正方形，保持置中。"""
    h, w = image_array.shape[:2]
    if h == w:
        return image_array
    size = max(h, w)
    pad_h_top = (size - h) // 2
    pad_h_bot = size - h - pad_h_top
    pad_w_left = (size - w) // 2
    pad_w_right = size - w - pad_w_left
    if image_array.ndim == 3:
        return np.pad(image_array,
                      ((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right), (0, 0)),
                      mode='constant')
    return np.pad(image_array,
                  ((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
                  mode='constant')


def prepare_for_medgemma(image_array: np.ndarray) -> Image.Image:
    """將 numpy 影像處理為 MedGemma 所需的 896x896 PIL Image。"""
    squared = pad_to_square(image_array)
    pil_img = Image.fromarray(squared)
    pil_img = pil_img.resize((MEDGEMMA_INPUT_SIZE, MEDGEMMA_INPUT_SIZE), Image.LANCZOS)
    return pil_img


def map_box_to_original(
    box_norm: List[float],
    crop_coords: List[int],
    original_hw: Tuple[int, int],
) -> List[float]:
    """
    將裁剪影像上的 0-1000 正規化座標映射回原始影像的像素座標。

    Args:
        box_norm:     [y0, x0, y1, x1] (0-1000 正規化座標, 在裁剪影像上)
        crop_coords:  [cy0, cx0, cy1, cx1] 裁剪區域在原始影像上的像素範圍
        original_hw:  (H, W) 原始影像尺寸

    Returns:
        [y0, x0, y1, x1] 原始影像上的像素座標
    """
    y0_n, x0_n, y1_n, x1_n = box_norm
    cy0, cx0, cy1, cx1 = crop_coords
    crop_h = cy1 - cy0
    crop_w = cx1 - cx0

    real_y0 = cy0 + (y0_n / COORD_NORM) * crop_h
    real_x0 = cx0 + (x0_n / COORD_NORM) * crop_w
    real_y1 = cy0 + (y1_n / COORD_NORM) * crop_h
    real_x1 = cx0 + (x1_n / COORD_NORM) * crop_w

    h, w = original_hw
    real_y0 = max(0, min(real_y0, h))
    real_x0 = max(0, min(real_x0, w))
    real_y1 = max(0, min(real_y1, h))
    real_x1 = max(0, min(real_x1, w))

    return [real_y0, real_x0, real_y1, real_x1]


def map_box_fullimage(
    box_norm: List[float],
    original_hw: Tuple[int, int],
) -> List[float]:
    """
    將全幅影像上的 0-1000 正規化座標映射為原始影像像素座標。
    (考慮 padding-to-square 的偏移)
    """
    y0_n, x0_n, y1_n, x1_n = box_norm
    h, w = original_hw
    size = max(h, w)
    pad_h = (size - h) / 2
    pad_w = (size - w) / 2

    real_y0 = (y0_n / COORD_NORM) * size - pad_h
    real_x0 = (x0_n / COORD_NORM) * size - pad_w
    real_y1 = (y1_n / COORD_NORM) * size - pad_h
    real_x1 = (x1_n / COORD_NORM) * size - pad_w

    real_y0 = max(0, min(real_y0, h))
    real_x0 = max(0, min(real_x0, w))
    real_y1 = max(0, min(real_y1, h))
    real_x1 = max(0, min(real_x1, w))

    return [real_y0, real_x0, real_y1, real_x1]


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """計算兩個 [y0, x0, y1, x1] bounding box 的 IoU。"""
    ya0, xa0, ya1, xa1 = box_a
    yb0, xb0, yb1, xb1 = box_b

    inter_y0 = max(ya0, yb0)
    inter_x0 = max(xa0, xb0)
    inter_y1 = min(ya1, yb1)
    inter_x1 = min(xa1, xb1)

    inter_area = max(0, inter_y1 - inter_y0) * max(0, inter_x1 - inter_x0)
    area_a = max(0, ya1 - ya0) * max(0, xa1 - xa0)
    area_b = max(0, yb1 - yb0) * max(0, xb1 - xb0)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_boxes_from_response(response_text: str) -> List[List[float]]:
    """
    從 MedGemma 的回應中解析 box_2d 座標。
    預期格式: [y0, x0, y1, x1] (0-1000 正規化座標)
    """
    boxes = []
    # 嘗試解析 JSON 格式
    json_patterns = re.findall(r'\{[^}]*"box_2d"\s*:\s*\[([^\]]+)\][^}]*\}', response_text)
    for pattern in json_patterns:
        try:
            coords = [float(x.strip()) for x in pattern.split(",")]
            if len(coords) == 4:
                boxes.append(coords)
        except ValueError:
            continue

    # 如果 JSON 解析失敗，嘗試直接找 list pattern
    if not boxes:
        list_patterns = re.findall(
            r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,'
            r'\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
            response_text,
        )
        for m in list_patterns:
            coords = [float(x) for x in m]
            if all(0 <= c <= 1000 for c in coords):
                boxes.append(coords)

    return boxes


def get_ground_truth_bbox(
    seg_data: np.ndarray,
    label_id: int,
    slice_idx: int,
) -> Optional[List[int]]:
    """
    從分割 mask 中取得指定器官在某切片上的 ground truth bounding box。

    Returns:
        [y0, x0, y1, x1] 像素座標，或 None
    """
    slice_mask = seg_data[:, :, slice_idx] == label_id
    if not slice_mask.any():
        return None
    rows = np.any(slice_mask, axis=1)
    cols = np.any(slice_mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(y0), int(x0), int(y1), int(x1)]


# ============================================================
# 2. 從 TotalSegmentator 官方資料擷取器官 ROI
# ============================================================

def load_totalseg_test_data(
    target_organ: str = "liver",
    ct_path: Optional[str] = None,
    seg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    載入 TotalSegmentator 官方測試資料，並擷取目標器官的 ROI。

    使用 test_data/example_ct_sm.nii.gz (CT) +
         test_data/example_seg.nii.gz  (參考分割)。

    Returns:
        dict 包含:
            full_slice_hu, cropped_hu, crop_coords, slice_idx,
            organ_name, label_id, gt_bbox, ct_shape
    """
    import nibabel as nib

    ct_file = ct_path or str(TEST_DATA_DIR / "example_ct_sm.nii.gz")
    seg_file = seg_path or str(TEST_DATA_DIR / "example_seg.nii.gz")

    logger.info(f"載入 CT:  {ct_file}")
    logger.info(f"載入 Seg: {seg_file}")

    ct_nii = nib.load(ct_file)
    seg_nii = nib.load(seg_file)
    ct_data = ct_nii.get_fdata().astype(np.float32)
    seg_data = seg_nii.get_fdata().astype(np.int32)

    logger.info(f"CT shape:  {ct_data.shape}, spacing: {ct_nii.header.get_zooms()}")
    logger.info(f"Seg shape: {seg_data.shape}")

    # 取得目標器官 label ID
    label_id = ORGAN_LABEL_MAP.get(target_organ)
    if label_id is None:
        available = [k for k, v in ORGAN_LABEL_MAP.items()
                     if (seg_data == v).any()]
        raise ValueError(
            f"未知器官 '{target_organ}'. "
            f"資料中可用的器官: {available}"
        )

    organ_mask = seg_data == label_id
    if not organ_mask.any():
        available = [LABEL_ID_TO_NAME.get(int(lid), f"label_{int(lid)}")
                     for lid in np.unique(seg_data) if lid > 0]
        raise ValueError(
            f"器官 '{target_organ}' (label={label_id}) 在分割資料中不存在. "
            f"可用器官: {available}"
        )

    voxel_count = organ_mask.sum()
    logger.info(f"目標器官: {target_organ} (label={label_id}), {voxel_count} voxels")

    # 選取器官最集中的 axial 切片
    organ_voxels = np.where(organ_mask)
    z_indices = organ_voxels[2]
    z_unique, z_counts = np.unique(z_indices, return_counts=True)
    z_best = z_unique[np.argmax(z_counts)]  # 選取器官體積最大的切片
    logger.info(f"選取切片 z={z_best} (該切片器官面積最大: {z_counts.max()} voxels)")

    full_slice_hu = ct_data[:, :, z_best]
    slice_mask = organ_mask[:, :, z_best]

    # 計算 Ground Truth Bounding Box
    gt_bbox = get_ground_truth_bbox(seg_data, label_id, z_best)
    logger.info(f"Ground Truth BBox: {gt_bbox}")

    # 計算裁剪 Bounding Box (加 15% padding)
    rows = np.any(slice_mask, axis=1)
    cols = np.any(slice_mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    h, w = full_slice_hu.shape
    pad_y = max(3, int((y1 - y0) * 0.15))
    pad_x = max(3, int((x1 - x0) * 0.15))
    crop_coords = [
        max(0, y0 - pad_y),
        max(0, x0 - pad_x),
        min(h, y1 + pad_y + 1),
        min(w, x1 + pad_x + 1),
    ]

    cropped_hu = full_slice_hu[
        crop_coords[0]:crop_coords[2],
        crop_coords[1]:crop_coords[3]
    ]

    logger.info(f"全幅切片尺寸: {full_slice_hu.shape}")
    logger.info(f"裁剪區域: {crop_coords}, 裁剪後尺寸: {cropped_hu.shape}")

    # 同時列出該切片上所有器官 (供參考)
    slice_labels = np.unique(seg_data[:, :, z_best])
    present_organs = [
        f"{LABEL_ID_TO_NAME.get(int(lid), f'id={int(lid)}')}"
        for lid in slice_labels if lid > 0
    ]
    logger.info(f"該切片上所有器官: {present_organs}")

    return {
        "full_slice_hu": full_slice_hu,
        "cropped_hu": cropped_hu,
        "crop_coords": crop_coords,
        "slice_idx": int(z_best),
        "organ_name": target_organ,
        "label_id": label_id,
        "gt_bbox": gt_bbox,
        "ct_shape": ct_data.shape,
        "seg_data": seg_data,
        "slice_mask": slice_mask,
    }


def load_3dircadb_data(
    ct_path: str,
    liver_mask_path: str,
    tumor_mask_path: str,
    slice_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    載入 3D-IRCADb NIfTI 資料 (含腫瘤 ground truth)。

    Args:
        ct_path:         CT volume (.nii.gz)
        liver_mask_path: Liver binary mask (.nii.gz)
        tumor_mask_path: Tumor mask (.nii.gz), 每個腫瘤不同 label
        slice_idx:       指定切片 (None = 自動選腫瘤最多的)

    Returns:
        dict 包含 full_slice_hu, cropped_hu, crop_coords, gt_tumor_bboxes, etc.
    """
    import nibabel as nib

    logger.info(f"載入 3D-IRCADb 資料:")
    logger.info(f"  CT:    {ct_path}")
    logger.info(f"  Liver: {liver_mask_path}")
    logger.info(f"  Tumor: {tumor_mask_path}")

    ct_data = nib.load(ct_path).get_fdata().astype(np.float32)
    liver_mask = nib.load(liver_mask_path).get_fdata().astype(np.uint8)
    tumor_mask = nib.load(tumor_mask_path).get_fdata().astype(np.uint8)

    logger.info(f"  CT shape: {ct_data.shape}")
    logger.info(f"  Liver voxels: {(liver_mask > 0).sum()}")
    logger.info(f"  Tumor voxels: {(tumor_mask > 0).sum()}")
    logger.info(f"  Tumor labels: {sorted(np.unique(tumor_mask[tumor_mask > 0]).tolist())}")

    # 選切片: 腫瘤面積最大
    if slice_idx is None:
        tumor_binary = tumor_mask > 0
        z_counts = tumor_binary.sum(axis=(0, 1))
        slice_idx = int(np.argmax(z_counts))
        logger.info(f"  自動選取切片 z={slice_idx} (腫瘤面積: {z_counts[slice_idx]} pixels)")

    full_slice_hu = ct_data[:, :, slice_idx]
    liver_slice = liver_mask[:, :, slice_idx] > 0
    tumor_slice = tumor_mask[:, :, slice_idx]

    # Liver bounding box → crop region (加 15% padding)
    if not liver_slice.any():
        raise ValueError(f"切片 z={slice_idx} 無肝臟區域")

    rows = np.any(liver_slice, axis=1)
    cols = np.any(liver_slice, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    h, w = full_slice_hu.shape
    pad_y = max(3, int((y1 - y0) * 0.15))
    pad_x = max(3, int((x1 - x0) * 0.15))
    crop_coords = [
        max(0, y0 - pad_y), max(0, x0 - pad_x),
        min(h, y1 + pad_y + 1), min(w, x1 + pad_x + 1),
    ]
    cropped_hu = full_slice_hu[crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]

    # 從腫瘤 mask 取得每個腫瘤的 GT bounding box (在原始像素座標)
    gt_tumor_bboxes = []
    tumor_labels = sorted(np.unique(tumor_slice[tumor_slice > 0]).tolist())
    for tid in tumor_labels:
        t_mask = tumor_slice == tid
        t_rows = np.any(t_mask, axis=1)
        t_cols = np.any(t_mask, axis=0)
        ty0, ty1 = np.where(t_rows)[0][[0, -1]]
        tx0, tx1 = np.where(t_cols)[0][[0, -1]]
        gt_tumor_bboxes.append({
            "label": f"tumor_{tid}",
            "bbox": [int(ty0), int(tx0), int(ty1), int(tx1)],
            "area": int(t_mask.sum()),
        })
        logger.info(f"  GT tumor_{tid}: bbox={[ty0,tx0,ty1,tx1]}, area={t_mask.sum()}")

    # 所有腫瘤的合併 GT bbox (for overall IoU)
    if gt_tumor_bboxes:
        all_ty0 = min(t["bbox"][0] for t in gt_tumor_bboxes)
        all_tx0 = min(t["bbox"][1] for t in gt_tumor_bboxes)
        all_ty1 = max(t["bbox"][2] for t in gt_tumor_bboxes)
        all_tx1 = max(t["bbox"][3] for t in gt_tumor_bboxes)
        gt_bbox = [all_ty0, all_tx0, all_ty1, all_tx1]
    else:
        gt_bbox = None

    logger.info(f"  全幅切片: {full_slice_hu.shape}")
    logger.info(f"  裁剪區域: {crop_coords}, 裁剪尺寸: {cropped_hu.shape}")
    logger.info(f"  GT 腫瘤數: {len(gt_tumor_bboxes)}")
    logger.info(f"  合併 GT bbox: {gt_bbox}")

    return {
        "full_slice_hu": full_slice_hu,
        "cropped_hu": cropped_hu,
        "crop_coords": crop_coords,
        "slice_idx": slice_idx,
        "organ_name": "liver",
        "label_id": 5,
        "gt_bbox": gt_bbox,                    # 合併所有腫瘤的 bbox
        "gt_tumor_bboxes": gt_tumor_bboxes,    # 每個腫瘤各自的 bbox
        "ct_shape": ct_data.shape,
        "seg_data": None,
        "slice_mask": liver_slice,
        "tumor_slice": tumor_slice,
        "mode": "detect_lesion",
    }


def run_totalsegmentator_live(ct_path: str, target_organ: str) -> Dict[str, Any]:
    """
    對自訂 CT 即時執行 TotalSegmentator 進行分割，然後擷取 ROI。
    (用於非官方測試資料的情況)
    """
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator

    logger.info(f"執行 TotalSegmentator 分割: {ct_path}")
    logger.info(f"目標器官: {target_organ}")

    # 執行 TotalSegmentator (使用 fast mode 節省記憶體/時間)
    mask_nifti = totalsegmentator(ct_path, task="total", fast=True, ml=True)
    mask_data = mask_nifti.get_fdata().astype(np.int32)

    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata().astype(np.float32)

    # 釋放 GPU 記憶體 (TotalSegmentator 用完)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 將分割結果暫存，然後用 load_totalseg_test_data 的邏輯處理
    # 先存暫時的 seg nifti
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        seg_tmp_path = f.name
        nib.save(mask_nifti, seg_tmp_path)

    result = load_totalseg_test_data(
        target_organ=target_organ,
        ct_path=ct_path,
        seg_path=seg_tmp_path,
    )

    os.unlink(seg_tmp_path)
    return result


def get_organ_mask_2d(image_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    對 2D 影像使用簡化的器官 ROI 策略。
    策略: Otsu threshold → 最大連通區域 → bounding box。
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    img = np.array(Image.open(image_path).convert("L")).astype(np.float32)
    logger.info(f"載入 2D 影像: {image_path}, 尺寸: {img.shape}")

    try:
        thresh = threshold_otsu(img)
        binary = img > thresh
    except ValueError:
        binary = img > img.mean()

    labeled = label(binary)
    regions = regionprops(labeled)

    if not regions:
        h, w = img.shape
        crop_coords = [int(h * 0.2), int(w * 0.2), int(h * 0.8), int(w * 0.8)]
    else:
        largest = max(regions, key=lambda r: r.area)
        y0, x0, y1, x1 = largest.bbox
        h, w = img.shape
        pad_y = int((y1 - y0) * 0.1)
        pad_x = int((x1 - x0) * 0.1)
        crop_coords = [
            max(0, y0 - pad_y), max(0, x0 - pad_x),
            min(h, y1 + pad_y), min(w, x1 + pad_x),
        ]

    cropped = img[crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
    logger.info(f"2D 裁剪區域: {crop_coords}, 裁剪後尺寸: {cropped.shape}")
    return img, cropped, crop_coords


# ============================================================
# 3. MedGemma 推論
# ============================================================

class MedGemmaInference:
    """MedGemma 1.5 推論封裝器。"""

    def __init__(self, model_id: str = MEDGEMMA_MODEL_ID, device_map: str = "auto"):
        logger.info(f"載入 MedGemma 模型: {model_id}")
        from transformers import pipeline as hf_pipeline

        self.pipe = hf_pipeline(
            "image-text-to-text",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": device_map,
            },
        )
        logger.info("MedGemma 模型載入完成")

    def run_inference(
        self,
        pil_image: Image.Image,
        object_name: str,
        additional_prompt: str = "",
        mode: str = "locate_organ",
    ) -> Dict[str, Any]:
        """
        對單張影像執行 MedGemma 推論。

        mode:
            "locate_organ"  — 定位器官 (原始行為)
            "detect_lesion" — 偵測病灶/腫瘤
        """
        if mode == "detect_lesion":
            prompt = (
                f"This is a CT image of the {object_name} region. "
                f"Detect all lesions, tumors, or abnormal masses visible in this image. "
                f"Output a JSON list where each element has a 'box_2d' key with "
                f"[y0, x0, y1, x1] coordinates normalized to 0-1000 and a 'label' key "
                f"describing the finding. "
                f"{additional_prompt}"
            )
        else:
            prompt = (
                f"Query: Where is the {object_name}? "
                f"Output a JSON list where each element has a 'box_2d' key with "
                f"[y0, x0, y1, x1] coordinates normalized to 0-1000. "
                f"{additional_prompt}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        logger.info(f"MedGemma 推論中... 目標: {object_name}")
        output = self.pipe(text=messages, max_new_tokens=500)
        raw_text = output[0]["generated_text"]

        # 如果 generated_text 是 list (chat format)，取最後一個 assistant 訊息
        if isinstance(raw_text, list):
            for msg in reversed(raw_text):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    raw_text = msg.get("content", "")
                    break
            else:
                raw_text = str(raw_text[-1]) if raw_text else ""

        boxes = parse_boxes_from_response(str(raw_text))
        logger.info(f"MedGemma 回應: {str(raw_text)[:300]}...")
        logger.info(f"解析到 {len(boxes)} 個 bounding boxes")

        return {"raw_response": str(raw_text), "boxes": boxes}


# ============================================================
# 4. 視覺化
# ============================================================

def draw_boxes_on_image(
    image: np.ndarray,
    boxes_pixel: List[List[float]],
    color: str = "red",
    label: str = "",
    linewidth: int = 3,
) -> Image.Image:
    """在影像上繪製 bounding boxes (像素座標)。"""
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image.copy()

    if rgb.max() > 255 or rgb.min() < 0:
        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    color_map = {
        "red": "#FF0000", "blue": "#0066FF", "green": "#00CC00",
        "yellow": "#FFCC00", "magenta": "#FF00FF", "cyan": "#00FFFF",
    }
    c = color_map.get(color, color)

    for i, box in enumerate(boxes_pixel):
        y0, x0, y1, x1 = box
        draw.rectangle([x0, y0, x1, y1], outline=c, width=linewidth)
        box_label = f"{label} #{i}" if label else f"#{i}"
        draw.text((x0 + 2, y0 + 2), box_label, fill=c)

    return pil_img


def visualize_ablation_results(
    data: Dict[str, Any],
    baseline_result: Dict[str, Any],
    organ_aware_result: Dict[str, Any],
    fair_iou_baseline: List[float],
    fair_iou_organ: List[float],
    pipeline_iou_baseline: List[float],
    pipeline_iou_organ: List[float],
    iou_cross: List[float],
    outputs_identical: bool,
    output_path: Path,
):
    """產生 ablation 比較的並排視覺化圖 (3x3 grid)。"""
    full_slice = data["full_slice_hu"]
    cropped_slice = data["cropped_hu"]
    crop_coords = data["crop_coords"]
    organ_name = data["organ_name"]
    gt_bbox = data["gt_bbox"]
    slice_mask = data["slice_mask"]
    original_hw = full_slice.shape[:2]

    fig, axes = plt.subplots(3, 3, figsize=(22, 20))
    fig.suptitle(
        f"Organ-Aware MedGemma 1.5 Ablation Study\n"
        f"Target: {organ_name} | Slice: z={data['slice_idx']} | "
        f"CT Shape: {data['ct_shape']}",
        fontsize=16, fontweight="bold",
    )

    # === Row 0: 輸入資料 ===
    full_rgb = hu_to_rgb(full_slice)

    # (0,0) 原始 CT (HU-to-RGB)
    axes[0, 0].imshow(full_rgb)
    axes[0, 0].set_title("Full CT Slice (HU-to-RGB)", fontsize=12)
    axes[0, 0].axis("off")

    # (0,1) 分割 Mask overlay
    mask_overlay = full_rgb.copy().astype(np.float32)
    organ_color = np.array([255, 100, 100], dtype=np.float32)
    mask_3d = np.stack([slice_mask] * 3, axis=-1).astype(np.float32)
    mask_overlay = (mask_overlay * (1 - 0.4 * mask_3d) + organ_color * 0.4 * mask_3d)
    mask_overlay = np.clip(mask_overlay, 0, 255).astype(np.uint8)
    axes[0, 1].imshow(mask_overlay)
    if gt_bbox:
        rect = patches.Rectangle(
            (gt_bbox[1], gt_bbox[0]), gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0],
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--',
        )
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f"TotalSegmentator Mask: {organ_name}", fontsize=12)
    axes[0, 1].axis("off")

    # (0,2) 裁剪影像
    cropped_rgb = hu_to_rgb(cropped_slice)
    axes[0, 2].imshow(cropped_rgb)
    cy0, cx0, cy1, cx1 = crop_coords
    axes[0, 2].set_title(
        f"Organ-Aware Crop [{cy0}:{cy1}, {cx0}:{cx1}]\n"
        f"Size: {cropped_slice.shape}",
        fontsize=12,
    )
    axes[0, 2].axis("off")

    # === Row 1: 偵測結果 ===
    # (1,0) Baseline 偵測
    baseline_boxes_px = [
        map_box_fullimage(b, original_hw) for b in baseline_result.get("boxes", [])
    ]
    vis_baseline = draw_boxes_on_image(full_rgb, baseline_boxes_px, color="red", label="BL")
    if gt_bbox:
        draw_bl = ImageDraw.Draw(vis_baseline)
        draw_bl.rectangle(
            [gt_bbox[1], gt_bbox[0], gt_bbox[3], gt_bbox[2]],
            outline="#00FF00", width=2,
        )
        draw_bl.text((gt_bbox[1], gt_bbox[0] - 12), "GT", fill="#00FF00")
    axes[1, 0].imshow(vis_baseline)
    axes[1, 0].set_title(
        f"[A] Baseline Detection ({len(baseline_boxes_px)} boxes)\n"
        f"Red=Predicted, Green=GT",
        fontsize=12,
    )
    axes[1, 0].axis("off")

    # (1,1) Organ-Aware 偵測 (mapped back)
    organ_boxes_px = [
        map_box_to_original(b, crop_coords, original_hw)
        for b in organ_aware_result.get("boxes", [])
    ]
    vis_organ = draw_boxes_on_image(full_rgb, organ_boxes_px, color="blue", label="OA")
    draw_oa = ImageDraw.Draw(vis_organ)
    draw_oa.rectangle([cx0, cy0, cx1, cy1], outline="#00CC00", width=2)
    draw_oa.text((cx0, cy0 - 12), "Crop Region", fill="#00CC00")
    if gt_bbox:
        draw_oa.rectangle(
            [gt_bbox[1], gt_bbox[0], gt_bbox[3], gt_bbox[2]],
            outline="#FFFF00", width=2,
        )
        draw_oa.text((gt_bbox[1], gt_bbox[0] - 12), "GT", fill="#FFFF00")
    axes[1, 1].imshow(vis_organ)
    axes[1, 1].set_title(
        f"[B] Organ-Aware Detection (mapped back)\n"
        f"Blue=Predicted, Yellow=GT, Green=Crop",
        fontsize=12,
    )
    axes[1, 1].axis("off")

    # (1,2) 兩者 overlay 比較
    vis_both = draw_boxes_on_image(full_rgb, baseline_boxes_px, color="red", label="BL")
    draw_both = ImageDraw.Draw(vis_both)
    for i, box in enumerate(organ_boxes_px):
        y0, x0, y1, x1 = box
        draw_both.rectangle([x0, y0, x1, y1], outline="#0066FF", width=3)
        draw_both.text((x0 + 2, y1 - 12), f"OA #{i}", fill="#0066FF")
    if gt_bbox:
        draw_both.rectangle(
            [gt_bbox[1], gt_bbox[0], gt_bbox[3], gt_bbox[2]],
            outline="#00FF00", width=2,
        )
    axes[1, 2].imshow(vis_both)
    axes[1, 2].set_title(
        "Overlay: Red=Baseline, Blue=OrganAware, Green=GT",
        fontsize=12,
    )
    axes[1, 2].axis("off")

    # === Row 2: 量化結果 ===
    # (2,0) Baseline 回應 + Fair IoU
    axes[2, 0].axis("off")
    bl_resp = baseline_result.get("raw_response", "N/A")
    bl_text = f"[A] Baseline Response:\n\n{bl_resp[:400]}"
    if fair_iou_baseline:
        bl_text += f"\n\nFair IoU (norm coords): {np.mean(fair_iou_baseline):.4f}"
    if pipeline_iou_baseline:
        bl_text += f"\nPipeline IoU (pixels):  {np.mean(pipeline_iou_baseline):.4f}"
    axes[2, 0].text(
        0.05, 0.95, bl_text,
        transform=axes[2, 0].transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#FFE0E0", alpha=0.8),
    )

    # (2,1) Organ-Aware 回應 + Fair IoU
    axes[2, 1].axis("off")
    oa_resp = organ_aware_result.get("raw_response", "N/A")
    oa_text = f"[B] Organ-Aware Response:\n\n{oa_resp[:400]}"
    if fair_iou_organ:
        oa_text += f"\n\nFair IoU (norm coords): {np.mean(fair_iou_organ):.4f}"
    if pipeline_iou_organ:
        oa_text += f"\nPipeline IoU (pixels):  {np.mean(pipeline_iou_organ):.4f}"
    axes[2, 1].text(
        0.05, 0.95, oa_text,
        transform=axes[2, 1].transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#E0E0FF", alpha=0.8),
    )

    # (2,2) 統計摘要表格 — 分兩層
    axes[2, 2].axis("off")
    summary_lines = [
        "══════════════════════════════════════",
        "    ABLATION STUDY METRICS SUMMARY    ",
        "══════════════════════════════════════",
        f"  Target: {organ_name} | z={data['slice_idx']}",
        f"  GT BBox: {gt_bbox}",
    ]

    if outputs_identical:
        summary_lines += [
            "",
            "  *** MODEL OUTPUTS IDENTICAL ***",
            "  Same normalized coords for both",
        ]

    summary_lines += [
        "",
        "  --- Fair IoU (Model Ability) ---",
        "  Compared in own norm-coord space",
        "  Removes mapping bias",
    ]
    bl_fair = f"{np.mean(fair_iou_baseline):.4f}" if fair_iou_baseline else "N/A"
    oa_fair = f"{np.mean(fair_iou_organ):.4f}" if fair_iou_organ else "N/A"
    summary_lines += [
        f"  Baseline:      {bl_fair}",
        f"  Organ-Aware:   {oa_fair}",
    ]

    summary_lines += [
        "",
        "  --- Pipeline IoU (System Effect) ---",
        "  Mapped back to original pixels",
        "  Includes mapping scale effect",
    ]
    bl_pipe = f"{np.mean(pipeline_iou_baseline):.4f}" if pipeline_iou_baseline else "N/A"
    oa_pipe = f"{np.mean(pipeline_iou_organ):.4f}" if pipeline_iou_organ else "N/A"
    summary_lines += [
        f"  Baseline:      {bl_pipe}",
        f"  Organ-Aware:   {oa_pipe}",
    ]

    # Conclusion
    summary_lines += [""]
    if outputs_identical:
        summary_lines.append("  CONCLUSION:")
        summary_lines.append("  Model outputs IDENTICAL.")
        summary_lines.append("  Pipeline IoU diff is from")
        summary_lines.append("  coordinate mapping geometry,")
        summary_lines.append("  NOT model localization skill.")
    elif fair_iou_baseline and fair_iou_organ:
        fair_bl = np.mean(fair_iou_baseline)
        fair_oa = np.mean(fair_iou_organ)
        if abs(fair_oa - fair_bl) < 0.02:
            summary_lines.append("  CONCLUSION: Similar ability")
        elif fair_oa > fair_bl:
            summary_lines.append(f"  CONCLUSION: Organ-Aware")
            summary_lines.append(f"  better (+{fair_oa - fair_bl:.4f})")
        else:
            summary_lines.append(f"  CONCLUSION: Baseline")
            summary_lines.append(f"  better (+{fair_bl - fair_oa:.4f})")

    summary_lines.append("======================================")

    axes[2, 2].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes[2, 2].transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#E8FFE8", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"視覺化結果已儲存: {output_path}")


# ============================================================
# 5. Ablation Test 主流程
# ============================================================

def run_ablation_test(
    data: Dict[str, Any],
    medgemma: MedGemmaInference,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    執行 ablation test: Baseline (Full) vs. Organ-Aware (Cropped)。
    支援兩種模式:
      - locate_organ:  定位器官 (原始)
      - detect_lesion: 偵測病灶/腫瘤 (新)
    """
    full_slice_hu = data["full_slice_hu"]
    cropped_hu = data["cropped_hu"]
    crop_coords = data["crop_coords"]
    organ_name = data["organ_name"]
    gt_bbox = data["gt_bbox"]
    original_hw = full_slice_hu.shape[:2]
    inference_mode = data.get("mode", "locate_organ")

    results = {"mode": inference_mode}

    if inference_mode == "detect_lesion":
        gt_tumor_bboxes = data.get("gt_tumor_bboxes", [])
        results["gt_tumor_bboxes"] = gt_tumor_bboxes
        logger.info(f"模式: 病灶偵測 | GT 腫瘤數: {len(gt_tumor_bboxes)}")

    # --- (A) Baseline: Full Image ---
    logger.info("=" * 60)
    logger.info("  [A] Baseline: Full Image Inference")
    logger.info("=" * 60)
    full_rgb = hu_to_rgb(full_slice_hu)
    full_pil = prepare_for_medgemma(full_rgb)
    full_pil.save(output_dir / "input_full.png")

    baseline_result = medgemma.run_inference(
        full_pil, organ_name, mode=inference_mode,
    )
    results["baseline"] = baseline_result

    # --- (B) Organ-Aware: Cropped Image ---
    logger.info("=" * 60)
    logger.info("  [B] Organ-Aware: Cropped Image Inference")
    logger.info("=" * 60)
    cropped_rgb = hu_to_rgb(cropped_hu)
    cropped_pil = prepare_for_medgemma(cropped_rgb)
    cropped_pil.save(output_dir / "input_cropped.png")

    organ_result = medgemma.run_inference(
        cropped_pil,
        organ_name,
        additional_prompt=(
            "This is a zoomed-in organ-aware crop. "
            "Focus on detecting lesions, tumors, or abnormal masses. Be precise."
            if inference_mode == "detect_lesion"
            else "This is a zoomed-in organ-aware crop from TotalSegmentator. Be precise."
        ),
        mode=inference_mode,
    )
    results["organ_aware"] = organ_result

    # --- (C) 座標映射 + IoU 計算 ---
    logger.info("=" * 60)
    logger.info("  [C] Computing IoU Metrics (Fair Evaluation)")
    logger.info("=" * 60)

    baseline_boxes_norm = baseline_result.get("boxes", [])
    organ_boxes_norm = organ_result.get("boxes", [])

    # 檢查模型輸出是否相同
    outputs_identical = (baseline_boxes_norm == organ_boxes_norm)
    if outputs_identical:
        logger.warning(
            "  ⚠ 模型對兩張影像回傳了相同的正規化座標！"
        )
        logger.warning(
            "  ⚠ IoU 差異完全來自座標映射 (coordinate mapping)，"
            "而非模型的定位能力差異。"
        )

    # --- 公平 IoU (Fair IoU): 在各自正規化座標系內比較 ---
    # 把 GT bbox 分別轉換到 Full-image 和 Crop-image 的 0-1000 座標系內，
    # 然後直接用正規化座標計算 IoU，消除座標映射帶來的偏差。

    # GT → Full Image 正規化座標
    gt_norm_full = None
    if gt_bbox:
        h, w = original_hw
        size = max(h, w)
        pad_h = (size - h) / 2
        pad_w = (size - w) / 2
        gt_norm_full = [
            (gt_bbox[0] + pad_h) / size * COORD_NORM,
            (gt_bbox[1] + pad_w) / size * COORD_NORM,
            (gt_bbox[2] + pad_h) / size * COORD_NORM,
            (gt_bbox[3] + pad_w) / size * COORD_NORM,
        ]
        logger.info(f"  GT in Full-image norm coords: {[f'{c:.1f}' for c in gt_norm_full]}")

    # GT → Crop Image 正規化座標
    gt_norm_crop = None
    if gt_bbox:
        cy0, cx0, cy1, cx1 = [float(c) for c in crop_coords]
        crop_h, crop_w = cy1 - cy0, cx1 - cx0
        # 把 GT 裁剪到 crop 範圍內，再轉換
        gt_y0_in_crop = max(0, gt_bbox[0] - cy0) / crop_h * COORD_NORM
        gt_x0_in_crop = max(0, gt_bbox[1] - cx0) / crop_w * COORD_NORM
        gt_y1_in_crop = min(crop_h, gt_bbox[2] - cy0) / crop_h * COORD_NORM
        gt_x1_in_crop = min(crop_w, gt_bbox[3] - cx0) / crop_w * COORD_NORM
        # 然後考慮 crop 影像也做了 pad_to_square + resize
        crop_size = max(crop_h, crop_w)
        crop_pad_h = (crop_size - crop_h) / 2
        crop_pad_w = (crop_size - crop_w) / 2
        gt_norm_crop = [
            (max(0, gt_bbox[0] - cy0) + crop_pad_h) / crop_size * COORD_NORM,
            (max(0, gt_bbox[1] - cx0) + crop_pad_w) / crop_size * COORD_NORM,
            (min(crop_h, gt_bbox[2] - cy0) + crop_pad_h) / crop_size * COORD_NORM,
            (min(crop_w, gt_bbox[3] - cx0) + crop_pad_w) / crop_size * COORD_NORM,
        ]
        logger.info(f"  GT in Crop-image norm coords:  {[f'{c:.1f}' for c in gt_norm_crop]}")

    # Fair IoU: Baseline 正規化座標 vs GT 正規化座標 (都在 full-image 空間)
    fair_iou_baseline = []
    if gt_norm_full:
        for i, box in enumerate(baseline_boxes_norm):
            iou = compute_iou(box, gt_norm_full)
            fair_iou_baseline.append(iou)
            logger.info(f"  [Fair] Baseline #{i} vs GT (norm): IoU={iou:.4f}")

    # Fair IoU: Organ-Aware 正規化座標 vs GT 正規化座標 (都在 crop-image 空間)
    fair_iou_organ = []
    if gt_norm_crop:
        for i, box in enumerate(organ_boxes_norm):
            iou = compute_iou(box, gt_norm_crop)
            fair_iou_organ.append(iou)
            logger.info(f"  [Fair] OrganAware #{i} vs GT (norm): IoU={iou:.4f}")

    # --- Pipeline IoU: 映射回原始座標後 vs GT (反映 pipeline 整體效果) ---
    baseline_boxes_px = [
        map_box_fullimage(b, original_hw) for b in baseline_boxes_norm
    ]
    organ_boxes_px = [
        map_box_to_original(b, crop_coords, original_hw)
        for b in organ_boxes_norm
    ]

    pipeline_iou_baseline = []
    if gt_bbox:
        for i, box in enumerate(baseline_boxes_px):
            iou = compute_iou(box, gt_bbox)
            pipeline_iou_baseline.append(iou)
            logger.info(f"  [Pipeline] Baseline #{i} vs GT (pixel): IoU={iou:.4f}")

    pipeline_iou_organ = []
    if gt_bbox:
        for i, box in enumerate(organ_boxes_px):
            iou = compute_iou(box, gt_bbox)
            pipeline_iou_organ.append(iou)
            logger.info(f"  [Pipeline] OrganAware #{i} vs GT (pixel): IoU={iou:.4f}")

    # Cross IoU
    iou_cross = []
    n_pairs = min(len(baseline_boxes_px), len(organ_boxes_px))
    for i in range(n_pairs):
        iou = compute_iou(baseline_boxes_px[i], organ_boxes_px[i])
        iou_cross.append(iou)

    results["outputs_identical"] = outputs_identical
    results["fair_iou_baseline"] = fair_iou_baseline
    results["fair_iou_organ"] = fair_iou_organ
    results["pipeline_iou_baseline"] = pipeline_iou_baseline
    results["pipeline_iou_organ"] = pipeline_iou_organ
    results["iou_cross"] = iou_cross
    results["baseline_boxes_pixel"] = baseline_boxes_px
    results["organ_aware_boxes_pixel"] = organ_boxes_px

    # --- (D) 視覺化 ---
    visualize_ablation_results(
        data, baseline_result, organ_result,
        fair_iou_baseline, fair_iou_organ,
        pipeline_iou_baseline, pipeline_iou_organ,
        iou_cross, outputs_identical,
        output_dir / "ablation_comparison.png",
    )

    # --- (D.5) 病灶偵測: 每個 GT tumor 的 per-tumor IoU ---
    per_tumor_baseline = []
    per_tumor_organ = []
    gt_tumor_bboxes = data.get("gt_tumor_bboxes", [])
    if inference_mode == "detect_lesion" and gt_tumor_bboxes:
        logger.info("=" * 60)
        logger.info("  [D.5] Per-tumor IoU Analysis")
        logger.info("=" * 60)

        for gt_t in gt_tumor_bboxes:
            gt_b = gt_t["bbox"]
            # Baseline: 找到與此 GT tumor 最匹配的預測 box
            best_bl_iou = 0.0
            for pred_b in baseline_boxes_px:
                iou = compute_iou(pred_b, gt_b)
                best_bl_iou = max(best_bl_iou, iou)
            per_tumor_baseline.append({"label": gt_t["label"], "best_iou": best_bl_iou})
            # Organ-Aware
            best_oa_iou = 0.0
            for pred_b in organ_boxes_px:
                iou = compute_iou(pred_b, gt_b)
                best_oa_iou = max(best_oa_iou, iou)
            per_tumor_organ.append({"label": gt_t["label"], "best_iou": best_oa_iou})
            logger.info(
                f"  {gt_t['label']} (area={gt_t['area']}): "
                f"BL best IoU={best_bl_iou:.4f}, OA best IoU={best_oa_iou:.4f}"
            )

        results["per_tumor_baseline"] = per_tumor_baseline
        results["per_tumor_organ"] = per_tumor_organ

    # --- (E) 儲存 JSON 報告 ---
    report = {
        "mode": inference_mode,
        "organ_name": organ_name,
        "label_id": data["label_id"],
        "slice_index": data["slice_idx"],
        "ct_shape": list(data["ct_shape"]),
        "original_slice_size": list(original_hw),
        "crop_coords": [int(c) for c in crop_coords],
        "ground_truth_bbox": gt_bbox,
        "outputs_identical": outputs_identical,
        "baseline": {
            "raw_response": baseline_result["raw_response"],
            "boxes_normalized": baseline_boxes_norm,
            "boxes_pixel": baseline_boxes_px,
            "fair_iou_vs_gt": fair_iou_baseline,
            "pipeline_iou_vs_gt": pipeline_iou_baseline,
        },
        "organ_aware": {
            "raw_response": organ_result["raw_response"],
            "boxes_normalized": organ_boxes_norm,
            "boxes_pixel": organ_boxes_px,
            "fair_iou_vs_gt": fair_iou_organ,
            "pipeline_iou_vs_gt": pipeline_iou_organ,
        },
        "analysis": {
            "fair_iou_baseline_avg": float(np.mean(fair_iou_baseline)) if fair_iou_baseline else None,
            "fair_iou_organ_avg": float(np.mean(fair_iou_organ)) if fair_iou_organ else None,
            "pipeline_iou_baseline_avg": float(np.mean(pipeline_iou_baseline)) if pipeline_iou_baseline else None,
            "pipeline_iou_organ_avg": float(np.mean(pipeline_iou_organ)) if pipeline_iou_organ else None,
            "cross_iou": iou_cross,
        },
    }

    if inference_mode == "detect_lesion":
        report["gt_tumor_bboxes"] = gt_tumor_bboxes
        report["per_tumor_baseline"] = per_tumor_baseline
        report["per_tumor_organ"] = per_tumor_organ
    report_path = output_dir / "ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Ablation 報告已儲存: {report_path}")

    # --- (F) 列印總結 ---
    print("\n" + "=" * 64)
    print(f"  ABLATION STUDY SUMMARY  [mode: {inference_mode}]")
    print("=" * 64)
    print(f"  Target Organ:          {organ_name}")
    print(f"  CT Shape:              {data['ct_shape']}")
    print(f"  Slice:                 z={data['slice_idx']}")
    print(f"  Ground Truth BBox:     {gt_bbox}")
    if inference_mode == "detect_lesion":
        print(f"  GT Tumors:             {len(gt_tumor_bboxes)}")
    print()

    bl_norm = baseline_boxes_norm
    oa_norm = organ_boxes_norm
    print(f"  [A] Baseline output (norm):    {bl_norm}")
    print(f"  [B] Organ-Aware output (norm): {oa_norm}")
    print(f"  [A] Baseline mapped (pixel):   {[f'[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]' for b in baseline_boxes_px]}")
    print(f"  [B] Organ-Aware mapped (px):   {[f'[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]' for b in organ_boxes_px]}")
    print()

    if outputs_identical:
        print("  ** MODEL OUTPUTS IDENTICAL **")
        print("  Pipeline IoU diff is from coordinate mapping, not model ability.")
        print()

    print("  -- Pipeline IoU (mapped to original pixels) --")
    p_bl = f"{np.mean(pipeline_iou_baseline):.4f}" if pipeline_iou_baseline else "N/A"
    p_oa = f"{np.mean(pipeline_iou_organ):.4f}" if pipeline_iou_organ else "N/A"
    print(f"  [A] Baseline:          {p_bl}")
    print(f"  [B] Organ-Aware:       {p_oa}")

    if not outputs_identical:
        print()
        print("  -- Fair IoU (in own normalized coords) --")
        f_bl = f"{np.mean(fair_iou_baseline):.4f}" if fair_iou_baseline else "N/A"
        f_oa = f"{np.mean(fair_iou_organ):.4f}" if fair_iou_organ else "N/A"
        print(f"  [A] Baseline:          {f_bl}")
        print(f"  [B] Organ-Aware:       {f_oa}")

    if inference_mode == "detect_lesion" and per_tumor_baseline:
        print()
        print("  -- Per-tumor best IoU --")
        print(f"  {'Tumor':<12s}  {'Baseline':>10s}  {'OrganAware':>10s}  {'Winner':<12s}")
        for bl_t, oa_t in zip(per_tumor_baseline, per_tumor_organ):
            w = "OA" if oa_t["best_iou"] > bl_t["best_iou"] else ("BL" if bl_t["best_iou"] > oa_t["best_iou"] else "Tie")
            print(f"  {bl_t['label']:<12s}  {bl_t['best_iou']:>10.4f}  {oa_t['best_iou']:>10.4f}  {w:<12s}")

    print()
    print(f"  Output:  {output_dir}")
    print("=" * 64 + "\n")

    return results


# ============================================================
# 6. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Organ-Aware MedGemma 1.5 Pipeline & Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用 TotalSegmentator 官方測試資料 — 器官定位 (預設)
  python organ_aware_pipeline.py

  # 病灶偵測模式 — 使用 3D-IRCADb 肝腫瘤資料
  python organ_aware_pipeline.py --lesion

  # 指定器官
  python organ_aware_pipeline.py --organ kidney_right

  # 使用自訂 CT (即時執行 TotalSegmentator)
  python organ_aware_pipeline.py --input /path/to/ct.nii.gz --organ liver

可用器官: spleen, kidney_right, kidney_left, gallbladder, liver,
          stomach, pancreas, heart, aorta, ...
        """,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="輸入影像路徑 (.nii.gz / .png / .jpg). 不指定則用內建測試資料")
    parser.add_argument("--seg", type=str, default=None,
                        help="預計算的分割 mask 路徑 (.nii.gz). 若提供則跳過 TotalSegmentator")
    parser.add_argument("--organ", type=str, default="liver",
                        help="目標器官 (預設: liver)")
    parser.add_argument("--lesion", action="store_true",
                        help="病灶偵測模式: 使用 3D-IRCADb 腫瘤資料，偵測 lesion 而非定位器官")
    parser.add_argument("--tumor-mask", type=str, default=None,
                        help="腫瘤 mask 路徑 (.nii.gz). 搭配 --lesion 使用")
    parser.add_argument("--liver-mask", type=str, default=None,
                        help="肝臟 mask 路徑 (.nii.gz). 搭配 --lesion 使用")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"輸出目錄 (預設: {OUTPUT_DIR})")
    parser.add_argument("--model", type=str, default=MEDGEMMA_MODEL_ID,
                        help=f"MedGemma 模型 ID (預設: {MEDGEMMA_MODEL_ID})")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 確定輸入來源 ===
    if args.lesion:
        # ─── 病灶偵測模式 ───
        ircadb_dir = TEST_DATA_DIR / "3dircadb1_1_nifti"
        ct_file = Path(args.input) if args.input else ircadb_dir / "ct.nii.gz"
        liver_file = Path(args.liver_mask) if args.liver_mask else ircadb_dir / "liver_mask.nii.gz"
        tumor_file = Path(args.tumor_mask) if args.tumor_mask else ircadb_dir / "tumor_mask.nii.gz"

        for f, name in [(ct_file, "CT"), (liver_file, "Liver mask"), (tumor_file, "Tumor mask")]:
            if not f.exists():
                logger.error(f"{name} 不存在: {f}")
                sys.exit(1)

        logger.info("=== 病灶偵測模式 (3D-IRCADb) ===")
        data = load_3dircadb_data(
            ct_path=str(ct_file),
            liver_mask_path=str(liver_file),
            tumor_mask_path=str(tumor_file),
        )

    elif args.input is None:
        # ─── 器官定位模式 (TotalSegmentator 測試資料) ───
        ct_file = TEST_DATA_DIR / "example_ct_sm.nii.gz"
        seg_file = TEST_DATA_DIR / "example_seg.nii.gz"

        if not ct_file.exists() or not seg_file.exists():
            logger.error(
                f"官方測試資料不存在！請先下載:\n"
                f"  mkdir -p {TEST_DATA_DIR}\n"
                f"  wget https://github.com/wasserth/TotalSegmentator/raw/master/"
                f"tests/reference_files/example_ct_sm.nii.gz -O {ct_file}\n"
                f"  wget https://github.com/wasserth/TotalSegmentator/raw/master/"
                f"tests/reference_files/example_seg.nii.gz -O {seg_file}"
            )
            sys.exit(1)

        logger.info("使用 TotalSegmentator 官方測試資料")
        data = load_totalseg_test_data(
            target_organ=args.organ,
            ct_path=str(ct_file),
            seg_path=str(seg_file),
        )

    else:
        # ─── 自訂輸入 ───
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"輸入檔案不存在: {input_path}")
            sys.exit(1)

        if input_path.suffix in (".gz", ".nii") or str(input_path).endswith(".nii.gz"):
            if args.seg:
                data = load_totalseg_test_data(
                    target_organ=args.organ,
                    ct_path=str(input_path),
                    seg_path=args.seg,
                )
            else:
                data = run_totalsegmentator_live(str(input_path), args.organ)
        else:
            full_hu, cropped_hu, crop_coords = get_organ_mask_2d(str(input_path))
            data = {
                "full_slice_hu": full_hu,
                "cropped_hu": cropped_hu,
                "crop_coords": crop_coords,
                "slice_idx": 0,
                "organ_name": args.organ,
                "label_id": ORGAN_LABEL_MAP.get(args.organ, -1),
                "gt_bbox": None,
                "ct_shape": full_hu.shape,
                "seg_data": None,
                "slice_mask": np.zeros_like(full_hu, dtype=bool),
            }

    # === 載入 MedGemma ===
    medgemma = MedGemmaInference(model_id=args.model)

    # === 執行 Ablation Test ===
    results = run_ablation_test(data, medgemma, output_dir)

    logger.info("Pipeline 完成！")
    return results


if __name__ == "__main__":
    main()
