# UniMedDiff
UniMedDiff is a universal framework for synthesizing anatomically accurate and pathologically diverse medical images from clinical reports. Trained on noise-filtered image–text pairs, UniMedDiff extracts and aligns concise text embeddings with visual features from lengthy reports, enabling pathology-aware guidance for medical image generation. It facilitates controllable generation by integrating reports and prior disease knowledge into the diffusion process, demonstrating versatility across datasets, tasks, and modalities. By bridging clinical reports and visual biomarkers, UniMedDiff pioneers reliable medical image synthesis for advanced diagnostics and research.

---


### Data Preparation

Compress and prepare the training and validation data:

```bash
python data_compression.py --split=train
python data_compression.py --split=val
python data_compression_vis.py
```

### Training
Train the unconditional or disease-conditioned diffusion models:

```bash
# Vanilla denoising diffusion training
python vanilla_denoising_process.py --config=configs/MIMIC_256_squeeze77.py

# Disease-conditioned diffusion training
python disease_injection.py --config=configs/MIMIC_256_squeeze77_control.py
```

### Testing & Image Generation
Generate synthetic chest X-ray images from clinical reports and disease class labels:

```bash
python image_generation.py \
  --config configs/MIMIC_256_squeeze77_control.py \
  --nnet_path weights/MIMIC_256_squeeze77_cls/200000.ckpt/nnet_ema.pth \
  --output_path gen_img/UniMedDiff \
  --input_path assets/inputs/test_report.txt \
  --input_cls_path assets/inputs/test_cls.txt \
  --config.nnet.depth=16
```
### Model Weights

Trained model weights can be downloaded from Baidu Cloud:

- **Link:** https://pan.baidu.com/s/1rMjFHKBfgE47dWLSrD24hA  
- **Access Code:** `asdf`





