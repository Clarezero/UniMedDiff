# Diff-CXR
Diff-CXR is a diffusion-based framework that generates high-fidelity chest X-rays from radiology reports. It combines disease knowledge with report content to guide the generation process, producing anatomically accurate and pathologically diverse images across 11 common lung diseases. By leveraging noise-filtered data and efficient text encoding, Diff-CXR offers a clinically meaningful solution for text-to-image synthesis in medical imaging.

---

### Environment Setup
We recommend using conda to manage your environment:

```bash
conda create -n diff_cxr python=3.12
conda activate diff_cxr

# Install PyTorch with CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required libraries
pip install -e ./repos/open_clip -e ./repos/timm
pip install einops requests tqdm pyyaml packaging regex opencv-python tensorboard ml-collections accelerate wandb spicy ftfy
```

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
  --output_path gen_img/diff_CXR \
  --input_path assets/inputs/test_report.txt \
  --input_cls_path assets/inputs/test_cls.txt \
  --config.nnet.depth=16
```
### Pretrained Weights
Pretrained weights can be downloaded from the following Baidu Cloud link:
Link: https://pan.baidu.com/s/1rMjFHKBfgE47dWLSrD24hA
Access Code: asdf