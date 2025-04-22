# BioMedClip finetune
python biomedclip_finetune.py

# data compression
python data_compression.py --split=train
python data_compression.py --split=val
python data_compression_vis.py 

# train
python vanilla_denoising_process.py  --config=configs/MIMIC_256_squeeze77.py 
python disease_injection.py --config=configs/MIMIC_256_squeeze77_control.py 

# test
python image_generation.py  --config configs/MIMIC_256_squeeze77_control.py --nnet_path weights/MIMIC_256_squeeze77_cls/200000.ckpt/nnet_ema.pth --output_path gen_img/diff_CXR --input_path assets/inputs/test_report.txt  --input_cls_path assets/inputs/test_cls.txt  --config.nnet.depth=16









