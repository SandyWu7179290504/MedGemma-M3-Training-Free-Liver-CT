# MedGemma-M3  
### Training-Free Spatial Precision Enhancement for Liver Lesion Detection

MedGemma-M3 is a training-free spatial refinement framework built on the HAI-DEF medical foundation model **MedGemma 1.5** for small liver lesion localization in abdominal CT images.

Although MedGemma demonstrates strong multimodal reasoning capability, we identify a critical spatial failure mode in dense CT slices: attention diffusion and hallucinated localization when small lesions are embedded within complex anatomical background.

MedGemma-M3 restructures the inference process without modifying pretrained weights. By integrating ROI-focused multi-crop scaling, Spatial-Aware Attention Steering (SAAS), and multi-scale NMS aggregation, the framework improves spatial precision while preserving the model’s zero-shot medical knowledge.

In quantitative evaluation, MedGemma-M3 improves IoU from **0.0098 to 0.3769 (38× increase)** and reduces hallucination from **100% to 0%**, demonstrating that HAI-DEF models can achieve high spatial reliability through structured inference engineering alone.

---

## Core Components

### ROI Multi-Crop
CT slices are cropped using a liver mask to constrain the spatial search space and improve the signal-to-noise ratio.

### Spatial-Aware Attention Steering (SAAS)
Inference-time attention modulation is applied within the Vision Encoder to steer attention toward organ-relevant regions, suppressing hallucinated predictions without updating model parameters.

### Multi-Scale NMS Aggregation
Predictions from multiple padded crops are projected back to the original coordinate system and consolidated using non-maximum suppression to produce stable final localization.

---

## Pipeline Overview
```text
Full CT
   │
   ├── Liver ROI Crop
   │
   ├── MedGemma (Vanilla)
   │
   └── MedGemma + SAAS
           │
     Multi-Scale NMS
           │
     Final Bounding Boxes
```

---

## Requirements

- Python ≥ 3.10  
- PyTorch  
- Access to MedGemma 1.5 via Hugging Face (license acceptance required)

Login to Hugging Face:

```bash
huggingface-cli login
```

## Run Inference
```bash
python saas_pipeline.py --crop-saas --output results
```

## Experimental Results

| Model | IoU | Hallucination Rate |
|-------|------|--------------------|
| Vanilla MedGemma | 0.0098 | 100% |
| MedGemma-M3 | **0.3769** | **0%** |

## License

MedGemma © Google — subject to official model usage terms.  
Dataset usage must comply with the original dataset license.
