import torch
import os
import shutil
import numpy as np
import libs.autoencoder
import libs.clip
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
from glob import glob
import einops
import cv2 as cv
import open_clip

from bio_con_model import MyCLIPEncoder, MyCLIPEncoder_squeeze77, MyCLIPSqueeze_GLv1, MyCLIPEncoder_squeeze128, MyCLIPEncoder_squeeze32, MyCLIPSqueeze_GLv1

import random
random.seed(42)

class MyDatabase(Dataset):
    """
    Custom Dataset for MIMIC chest X-ray images and their corresponding text captions.

    Attributes:
        root: Root directory of the dataset.
        mode: Dataset split ('train' or 'val').
        size: Image resize dimension (height = width = size).
        paths: List of image file paths.
        cls: List of class labels for each image.
    """
    def __init__(self, root, mode, size=None):
        self.root = root
        self.height = self.width = size
        self.size = size
        self.paths = []
        self.cls = []
        self.mode = mode
        
        with open(root + f'/assets/inputs/{self.mode}_frontal_cls.txt', 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                img_path = os.path.join('assets/datasets/MIMIC',ann.split('&&')[0].strip()[77:])
                cls_info = ann.split('&&')[1].strip().split(' ')
                self.paths.append(img_path)
                self.cls.append(cls_info)
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        cls_info = self.cls[index]
        txt_name = img_path[21:].replace('/', '_').replace('jpg', 'txt')
        with open(f'assets/datasets/captions/{txt_name}', 'r') as f:
            target = f.readlines()
        target = [ann.strip() for ann in target]

        for step, con in enumerate(target):
            tokens = con.split(' ')
            target_len = len(tokens)
            if target_len > 256:
                diff = target_len - 256
                target[step] = ' '.join(tokens[diff:])

        image = cv.cvtColor(cv.resize(cv.imread(img_path), (self.size, self.size)), cv.COLOR_BGR2RGB)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        return image, target, cls_info

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    print(args)

    if args.split == "train":
        datas = MyDatabase(root='./', mode='train', size=resolution)
        save_dir = f'assets/datasets/MIMIC{resolution}_Ori_256Bio/train'
    elif args.split == "val":
        datas = MyDatabase(root='./', mode='val', size=resolution)
        save_dir = f'assets/datasets/MIMIC{resolution}_Ori_256Bio/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    print('Autoencoder loaded...')
    
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    model = MyCLIPEncoder_squeeze77(backbone)
    path = 'weights/Contrastive_Pretrain_v1_CL_77_seed3407/15-2.2943-2.0637.pth'
    model.load_state_dict(torch.load(path), strict=False)

    print('Clip text encoder loaded...')
    model.eval()
    model.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions, cls_info = data 
  
            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x,dtype=torch.float32, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()

            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            texts = tokenizer(captions, context_length=256)
            latent = model(texts.to(device))

            class_noun = ['There is atelectasis.', 'There is cardiomegaly.', 'There is consolidation.', 'There is edema.', 
                            'There is enlarged cardiomediastinum.', 'There is fracture.', 'There is lung lesion.', 'There is lung opacity.', 
                            'There is pleural effusion.', 'There is pneumonia.', 'There is pneumothorax.']

            class_prior = None
            index = 0

            for step, info in enumerate(cls_info):
                if step == 8 or step == 10 or step ==  13:
                    continue

                if info == '1':
                    if class_prior is None:
                        class_prior = class_noun[index]
                    else:
                        class_prior = class_prior + ' ' + class_noun[index]
                index = index + 1
            
            if class_prior is None:
                class_prior = 'There is no finding.'
            
            class_texts = tokenizer([class_prior], context_length=256)
            class_latent = model(class_texts.to(device))

            for i in range(len(class_latent)):
                c = class_latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_prior.npy'), c)

            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)


if __name__ == '__main__':
    main()
