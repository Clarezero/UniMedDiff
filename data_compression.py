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
    def __init__(self, root, mode, size=None):
        # from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size
        self.size = size
        # self.ids = glob(self.root + '/caption/*')
        self.paths = []
        self.cls = []
        self.mode = mode
        with open(root + f'/assets/inputs/{self.mode}_frontal_cls.txt', 'r', encoding='utf-8') as f:
        # with open(root + f'/CL_{self.mode}_frontal_1024.txt', 'r', encoding='utf-8') as f:
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

        ### token process
        for step, con in enumerate(target):
            tokens = con.split(' ')
            target_len = len(tokens)
            if target_len > 256:
                diff = target_len - 256
                target[step] = ' '.join(tokens[diff:])
        ###只保留最后的256个token

        # for step, con in enumerate(target):
        #     print(con) 
        # print(len(target))
        # exit()
        # image = self._load_image(key)
        image = cv.cvtColor(cv.resize(cv.imread(img_path), (self.size, self.size)), cv.COLOR_BGR2RGB)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        # target = []
        # target.append(ann)
        # label[10] = int(cls_info[11])
        return image, target, cls_info
        # return image, target

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MyDatabase(root='./', mode='train',
                             size=resolution)
        save_dir = f'assets/datasets/MIMIC{resolution}_Ori_256Bio/train'
    elif args.split == "val":
        datas = MyDatabase(root='./', mode='val',
                             size=resolution)
        save_dir = f'assets/datasets/MIMIC{resolution}_Ori_256Bio/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    # autoencoder = libs.autoencoder.get_model('/storage/ScientificPrograms/Diffusion/code/autoencoder_pretrain/pretrain_v2/Jul15_06-05-57/4-0.0602.pth')
    # autoencoder = libs.autoencoder.get_model('/storage/ScientificPrograms/Diffusion/code/autoencoder_pretrain/pretrain_v1/Jun30_06-34-55/1-0.0008.pth')
    autoencoder.to(device)
    print('Autoencoder loaded...')
    # clip = libs.clip.FrozenCLIPEmbedder()
    # clip.eval()
    # clip.to(device)
    # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    ###
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    
    # model = MyCLIPSqueeze_GLv1(backbone)
    # model = MyCLIPEncoder_squeeze32(backbone)
    model = MyCLIPEncoder_squeeze77(backbone)
    # Silu
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Contrastive_Pretrain_Squeezev1_77/Aug16_09-07-16/18-2.2396.pth'
    ### tmp 01 RELU
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Contrastive_Pretrain_Squeezev1_77/temp010/Aug23_09-07-12/16-2.2101.pth'
    # global and local alignment
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/CLData/Contrastive_Pretrain_v1/temp007/Oct08_03-09-51/15-2.5294-2.6787.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare/Contrastive_Pretrain_v1_CL_Attn1_seed3407/temp007/Oct16_14-16-34/15-2.1926-1.877.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare/Contrastive_Pretrain_v1_CL_256_seed3407_NoMultiView/temp007/Oct25_06-12-48/13-2.2027-1.9773.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare/Contrastive_Pretrain_v1_CL_Attn1_NoMultiView_seed3407/temp007/Oct17_01-16-27/12-2.1906-1.9689.pth'    
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare/Contrastive_Pretrain_v1_CL_Attn128_seed3407/temp007/Oct17_07-12-13/17-2.2606-1.9444.pth'    
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare/Contrastive_Pretrain_v1_CL_Attn32_seed3407/temp007/Oct17_07-12-32/15-2.1691-1.9031.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_77_seed3407/temp007/Nov26_09-44-48/15-2.2943-2.0637.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_128_seed3407/temp007/Dec02_13-31-33/13-2.2826-2.0873.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_32_seed3407/temp007/Dec02_13-31-07/17-2.3108-2.0872.pth'    
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_77_seed3407_LocalConstraint/temp007/loss_1_1/Jan08_12-43-09/13-2.3231-2.0601.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_32_seed3407_LocalConstraint/temp007/loss_1_1/Jan09_11-10-38/14-2.2763-2.0362.pth'
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/Compare_116/Contrastive_Pretrain_v1_CL_77_seed3407_LocalConstraintLR/temp007/loss_1_02/Jan16_13-22-16/12-2.2178-1.9988.pth'
    path = 'weights/Contrastive_Pretrain_v1_CL_77_seed3407/15-2.2943-2.0637.pth'
    model.load_state_dict(torch.load(path), strict=False)

    ###
    print('Clip text encoder loaded...')
    model.eval()
    model.to(device)
    # tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    ###

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions, cls_info = data 
  
            # x, captions = data  

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x,dtype=torch.float32, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()

            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            # np.save(os.path.join(save_dir, f'{idx}.npy'), moments)
            # shutil.copy(f'/storage/ScientificPrograms/Diffusion/code/U-ViT-main/assets/datasets/MIMIC256_Squeeze77_CL_Filter/train/{idx}.npy', os.path.join(save_dir, f'{idx}.npy'))
            ### text latent
            texts = tokenizer(captions, context_length=256)
            latent = model(texts.to(device))

            ### text prior
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
                # p = image_prior.detach().cpu().numpy()
                # np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), np.concatenate((c, p), axis=0))
        
            # # latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                # p = image_prior.detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)


if __name__ == '__main__':
    main()
