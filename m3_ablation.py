"""M3 Ablation Study: A=Full+Vanilla, B=Crop+Vanilla, C=Crop+SAAS, D=M3 (full)."""
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def run_m3_ablation_study(
    data: Dict[str, Any],
    medgemma,
    output_dir: Path,
    best_layer_range: Tuple[int, int] = (14, 26),
    alpha: float = 1e4,
) -> Dict[str, Dict[str, Any]]:
    """
    Ablation study: decompose M3 into 3 components.
    A) Full + Vanilla     — no crop, no SAAS (naive)
    B) Crop + Vanilla     — crop only, no SAAS
    C) Crop + SAAS        — crop + SAAS, single scale
    D) Crop + SAAS + Multi-scale NMS — full M3
    """
    from saas_pipeline import (
        prepare_for_medgemma, hu_to_rgb, map_box_fullimage, filter_hallucinations,
        _eval_boxes_to_metrics, run_baseline_crop_saas, run_m3_with_config,
        LESION_DETECTION_PROMPT, CROPPED_LESION_PROMPT,
        M3_DEFAULT_PADDING_RATIOS,
    )
    import logging
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_slice_hu = data["full_slice_hu"]
    liver_mask = data["liver_slice_mask"]
    crop_coords = data["crop_coords"]
    gt_tumor_bboxes = data["gt_tumor_bboxes"]
    gt_bbox = data["gt_bbox"]
    h, w = full_slice_hu.shape[:2]
    cy0, cx0, cy1, cx1 = crop_coords
    results = {}

    # A: Full + Vanilla
    logger.info("=" * 60)
    logger.info("  [Ablation A] Full image + Vanilla (no crop, no SAAS)")
    logger.info("=" * 60)
    full_pil = prepare_for_medgemma(hu_to_rgb(full_slice_hu))
    r_a = medgemma.run_inference(
        full_pil, LESION_DETECTION_PROMPT,
        seg_mask_896=None, steering_mode="vanilla",
        max_new_tokens=1024, temperature=0.3,
    )
    filtered_a, _ = filter_hallucinations(
        r_a["boxes"], organ_mask_2d=liver_mask, original_hw=(h, w),
    )
    boxes_a = [map_box_fullimage(b, (h, w)) for b in filtered_a]
    r_a_metrics = _eval_boxes_to_metrics(boxes_a, gt_tumor_bboxes, gt_bbox)
    r_a_metrics["label"] = "A: Full+Vanilla"
    results["A_full_vanilla"] = r_a_metrics
    logger.info(f"  IoU={r_a_metrics['best_iou_vs_gt']:.4f}, pred={r_a_metrics['n_predicted']}, halluc={r_a_metrics['n_hallucinated']}")

    # B: Crop + Vanilla
    logger.info("=" * 60)
    logger.info("  [Ablation B] Crop + Vanilla (no SAAS)")
    logger.info("=" * 60)
    crop_hu = full_slice_hu[cy0:cy1, cx0:cx1]
    crop_liver = liver_mask[cy0:cy1, cx0:cx1]
    crop_hw = crop_hu.shape[:2]
    crop_pil = prepare_for_medgemma(hu_to_rgb(crop_hu))
    r_b = medgemma.run_inference(
        crop_pil, CROPPED_LESION_PROMPT,
        seg_mask_896=None, steering_mode="vanilla",
        max_new_tokens=1024, temperature=0.3,
    )
    filtered_b, _ = filter_hallucinations(
        r_b["boxes"], organ_mask_2d=crop_liver, original_hw=crop_hw,
    )
    boxes_b = [[b[0]+cy0, b[1]+cx0, b[2]+cy0, b[3]+cx0]
               for b in (map_box_fullimage(x, crop_hw) for x in filtered_b)]
    r_b_metrics = _eval_boxes_to_metrics(boxes_b, gt_tumor_bboxes, gt_bbox)
    r_b_metrics["label"] = "B: Crop+Vanilla"
    results["B_crop_vanilla"] = r_b_metrics
    logger.info(f"  IoU={r_b_metrics['best_iou_vs_gt']:.4f}, pred={r_b_metrics['n_predicted']}, halluc={r_b_metrics['n_hallucinated']}")

    # C: Crop + SAAS
    logger.info("=" * 60)
    logger.info("  [Ablation C] Crop + SAAS (single scale)")
    logger.info("=" * 60)
    r_c = run_baseline_crop_saas(data, medgemma, crop_coords, best_layer_range, alpha)
    r_c["label"] = "C: Crop+SAAS"
    results["C_crop_saas"] = r_c
    logger.info(f"  IoU={r_c['best_iou_vs_gt']:.4f}, pred={r_c['n_predicted']}, halluc={r_c['n_hallucinated']}")

    # D: M3 full
    logger.info("=" * 60)
    logger.info("  [Ablation D] Crop + SAAS + Multi-scale NMS (M3)")
    logger.info("=" * 60)
    r_d = run_m3_with_config(
        data, medgemma, crop_coords, best_layer_range, alpha,
        padding_ratios=M3_DEFAULT_PADDING_RATIOS,
        temperature=0.3,
    )
    r_d["label"] = "D: M3 (full)"
    results["D_m3_full"] = r_d
    logger.info(f"  IoU={r_d['best_iou_vs_gt']:.4f}, pred={r_d['n_predicted']}, halluc={r_d['n_hallucinated']}")

    # Visualization
    visualize_m3_ablation(data, results, output_dir)

    # Report
    report = {
        "experiment": "m3_ablation_study",
        "gt_bbox": gt_bbox,
        "gt_tumor_bboxes": gt_tumor_bboxes,
        "scores": {k: {"best_iou_vs_gt": v["best_iou_vs_gt"], "n_predicted": v["n_predicted"],
                      "n_hallucinated": v["n_hallucinated"], "hallucination_rate": v["hallucination_rate"]}
                     for k, v in results.items()},
    }
    with open(output_dir / "m3_ablation_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("  M3 ABLATION STUDY — Crop | SAAS | Multi-scale NMS")
    print("=" * 70)
    print(f"  GT bbox: {gt_bbox} | Tumors: {len(gt_tumor_bboxes)}")
    print()
    print(f"  {'Config':<28s} {'IoU':>8s} {'Pred':>6s} {'Halluc':>7s} {'Rate':>6s}")
    print("  " + "-" * 58)
    for name in ("A_full_vanilla", "B_crop_vanilla", "C_crop_saas", "D_m3_full"):
        r = results[name]
        print(f"  {r['label']:<28s} {r['best_iou_vs_gt']:>8.4f} {r['n_predicted']:>6d} {r['n_hallucinated']:>7d} {r['hallucination_rate']:>5.0%}")
    print()
    print(f"  Output: {output_dir}")
    print("=" * 70 + "\n")

    return results


def visualize_m3_ablation(data, results, output_dir):
    """2×2 grid + bar chart for M3 ablation."""
    from saas_pipeline import hu_to_rgb
    full_rgb = hu_to_rgb(data["full_slice_hu"])
    crop_coords = data["crop_coords"]
    gt_tumor_bboxes = data["gt_tumor_bboxes"]
    cy0, cx0, cy1, cx1 = crop_coords

    step_order = ("A_full_vanilla", "B_crop_vanilla", "C_crop_saas", "D_m3_full")
    colors = {"A_full_vanilla": "#FF4444", "B_crop_vanilla": "#FF8800",
              "C_crop_saas": "#4488FF", "D_m3_full": "#00AA66"}
    titles = {
        "A_full_vanilla": "A: Full + Vanilla",
        "B_crop_vanilla": "B: Crop + Vanilla",
        "C_crop_saas": "C: Crop + SAAS",
        "D_m3_full": "D: M3 (Crop+SAAS+Multi-scale NMS)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    fig.suptitle(
        "M3 Ablation Study — Contribution of Crop, SAAS, Multi-scale NMS\n"
        "Yellow = GT | Colored = Predicted",
        fontsize=13, fontweight="bold",
    )
    for idx, name in enumerate(step_order):
        ax = axes[idx // 2, idx % 2]
        r = results[name]
        color = colors.get(name, "#888")
        pil_vis = Image.fromarray(full_rgb.copy())
        draw = ImageDraw.Draw(pil_vis)
        draw.rectangle([cx0, cy0, cx1, cy1], outline="#00FFFF", width=2)
        for gt_t in gt_tumor_bboxes:
            b = gt_t["bbox"]
            draw.rectangle([b[1], b[0], b[3], b[2]], outline="#FFFF00", width=3)
        for i, box in enumerate(r.get("boxes_full_coords", [])):
            y0, x0, y1, x1 = [int(round(v)) for v in box]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            try:
                draw.text((x0 + 2, y0 + 2), f"P{i}", fill=color)
            except Exception:
                pass
        ax.imshow(pil_vis)
        ax.set_title(
            f"{titles.get(name, name)}\n"
            f"IoU={r['best_iou_vs_gt']:.4f} | Pred={r['n_predicted']} | Halluc={r['n_hallucinated']} ({r['hallucination_rate']:.0%})",
            fontsize=10, fontweight="bold", color=color,
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "m3_ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    labels = [results[n]["label"] for n in step_order]
    ious = [results[n]["best_iou_vs_gt"] for n in step_order]
    preds = [results[n]["n_predicted"] for n in step_order]
    hallucs = [results[n]["n_hallucinated"] for n in step_order]
    x = np.arange(len(labels))
    w = 0.25
    ax2.bar(x - w, ious, w, label="IoU", color="#0066CC")
    ax2.bar(x, preds, w, label="Predicted", color="#00AA66")
    ax2.bar(x + w, hallucs, w, label="Hallucinated", color="#CC0000")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend()
    ax2.set_ylabel("Value")
    ax2.set_title("M3 Ablation — IoU, Predicted, Hallucinated per Config")
    plt.tight_layout()
    plt.savefig(output_dir / "m3_ablation_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
