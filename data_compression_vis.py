import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm

import open_clip

import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F


from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

class MyCLIPEncoder_squeeze77(nn.Module):
    """
    A custom CLIP text encoder that compresses token-level features
    using 1D convolution layers.
    """
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze77, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        
    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]

        return text_features


def main():
   
    prompts = [
        'There is unchanged bibasal atelectasis, infection cannot be excluded.',
        'The lungs are well inflated with bibasilar linear atelectasis. ',
        'There is some atelectasis at the lung bases.',
        'Moderate cardiomegaly persists.',
        'Moderate cardiomegaly is noted.',
        'Persistent moderate cardiomegaly.',
        'This is compatible but not diagnostic of a focal consolidation.',
        'There is increasing consolidation in the right lower lung.',
        'There is consolidation.',
        'Pulmonary edema is present.',
        'There is edema.',
        'Edema is noted.',
        'Enlarged cardiomediastinum is present.',
        'There is enlarged cardiomediastinum.',
        'Enlarged cardiomediastinum is noted.',
        'There is fracture.',
        'Fracture is noted.',
        'Fracture is present.',
        'There is Lung Opacity.',
        'Lung Opacity is noted.',
        'Lung Opacity is present.',
        'There is lung lesion.',
        'Lung lesion is noted.',
        'Lung lesion is present.',
        'There is pleural effusion.',
        'Pleural effusion is noted.',
        'Pleural effusion is present.',
        'There is pneumonia.',
        'Pneumonia is noted.',
        'Pneumonia is present.',
        'There is Pneumothorax.',
        'Pneumothorax is noted.',
        'Pneumothorax is present.',
    ]

    device = 'cuda'

    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    model = MyCLIPEncoder_squeeze77(backbone)
    state_dict = torch.load('weights/Contrastive_Pretrain_v1_CL_77_seed3407/15-2.2943-2.0637.pth')
    load_info = model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(torch.load('weights/Contrastive_Pretrain_v1_CL_77_seed3407/15-2.2943-2.0637.pth'),strict=False)

    print('Clip text encoder loaded...')

    model.eval()
    model.to(device)

    save_dir = f'vis_template'
    os.makedirs(save_dir, exist_ok=True)
    texts = tokenizer(prompts, context_length=256)

    latent = model(texts.to(device))
    for i in range(len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), {'prompt':prompts[i], 'latent':c})
    print(latent[0].detach().cpu().numpy().shape)

if __name__ == '__main__':
    main()
