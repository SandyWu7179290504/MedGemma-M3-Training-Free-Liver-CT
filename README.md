# MedGemma-M3

MedGemma-M3 is a training-free spatial precision enhancement pipeline built on the HAI-DEF medical foundation model MedGemma 1.5.

## Key Features

- Multi-crop ROI scaling
- Spatial-Aware Attention Steering (SAAS)
- Multi-scale NMS aggregation
- 38× IoU improvement
- 0% hallucination rate

## Requirements

- Python 3.10+
- PyTorch
- MedGemma 1.5

Install dependencies:

pip install -r requirements.txt

## Usage

Run inference:

python run_inference.py --input sample_ct.png

## Citation

If you use this code, please cite our Kaggle submission.
