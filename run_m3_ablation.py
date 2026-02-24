#!/usr/bin/env python3
"""Run M3 ablation study. Usage: python run_m3_ablation.py [--output saas_m3_ablation]"""
import sys
from pathlib import Path

# Add parent for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from saas_pipeline import load_3dircadb_data, SAASMedGemma, TEST_DATA_DIR
from m3_ablation import run_m3_ablation_study

def main():
    args = sys.argv[1:]
    output_dir = Path("saas_m3_ablation")
    for i, a in enumerate(args):
        if a == "--output" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            break

    ircadb_dir = TEST_DATA_DIR / "3dircadb1_1_nifti"
    ct_path = str(ircadb_dir / "ct.nii.gz")
    liver_path = str(ircadb_dir / "liver_mask.nii.gz")
    tumor_path = str(ircadb_dir / "tumor_mask.nii.gz")

    data = load_3dircadb_data(ct_path, liver_path, tumor_path)
    medgemma = SAASMedGemma(model_id="google/medgemma-1.5-4b-it")
    run_m3_ablation_study(data, medgemma, output_dir, best_layer_range=(14, 26), alpha=1e4)

if __name__ == "__main__":
    main()
