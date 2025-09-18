import os
import json
import random
import logging

import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime
from tools.utils import get_root_logger

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCEWithLogitsLoss

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from tools.loss import MyCLIP, MyCLIPSqueeze_v1, MyCLIPSqueeze_v4, MyCLIPSqueeze_GLv1
from PIL import Image
import open_clip
import einops
from tools.clip_loss import ClipLoss

import torch.distributed as dist
import multiprocessing


backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


class XrayDataset(data.Dataset):
    """
    Custom dataset for loading X-ray images and corresponding text captions.
    """
    def __init__(
        self, infos, transform, mode):
        self.transform = transform
        self.mode = mode
        self.size = 224
        self.infos = infos

    def __getitem__(self, index):

        info = self.infos[index]
        image = cv.cvtColor(cv.resize(cv.imread(info), (224, 224)), cv.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        txt_name = info[22:].replace('/', '_').replace('jpg', 'txt')
        with open(f'assets/datasets/captions/{txt_name}', 'r') as f:
            target = f.readlines()
        target = [ann.strip() for ann in target]
        tokens = target[0].split(' ')
        target_len = len(tokens)

        if target_len > 256:
            diff = target_len - 256
            target = [' '.join(tokens[diff:])]
        
        token_embeddings = tokenizer(target, context_length=256)
        return  {'image':image, 'text':token_embeddings}

    def __len__(self):
        return len(self.infos)
        

if __name__ == '__main__':
    multiprocessing.freeze_support()

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    seed_number = 3407
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    np.random.seed(seed_number)


    log_dir = f"Compare_INFO_Squeeze/Contrastive_Pretrain_v1_CL_77_seed3407/temp007/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/logger.log'
    logger = get_root_logger(name='TextEncoder_Pretrain_test', log_level=logging.DEBUG, log_file=log_file)
    writer = SummaryWriter(log_dir=log_dir)


    
        
    model = MyCLIPSqueeze_v1(backbone)
    model.to(device)
    logger.info('Model and weights successfully loaded...')

        
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))
    logger.info('Model successfully loaded and backbone freezed...')


    infos = []
    with open('assets/inputs/train_frontal.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            infos.append(ann.strip())

    val_infos = []
    with open('assets/inputs/val_frontal.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            val_infos.append(ann.strip())

    random.shuffle(infos)

    test_infos = []
    with open('assets/inputs/test_frontal.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            test_infos.append(ann.strip())

    logger.info(f'{len(infos)} training data samples have been collected...')
    logger.info(f'{len(val_infos)} validation data samples have been collected...')
    logger.info(f'{len(test_infos)} testing data samples have been collected...')


    

    train_dataset = XrayDataset(infos, None, 'train')
    val_dataset = XrayDataset(val_infos, None, 'val')
    test_dataset = XrayDataset(test_infos, None, 'val')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True,
                                        num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False,
                                        num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False,
                                        num_workers=8, pin_memory=True, persistent_workers=True)

    criterion = MSELoss()
    cls_criterion = BCEWithLogitsLoss()
    clip_loss = ClipLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    epochs = 30
    ttl_iters = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)

    use_amp = True
    logit = torch.tensor(1 / 0.07).to(device)

    if use_amp:
        scaler = GradScaler()

    for epoch in range(epochs):
        train_losses = []

        val_losses = []
        test_losses = []
        val_recon_loss = []
        val_cls_loss = []

        model.train()
        model.train()
        for step, batch in tqdm(enumerate(train_loader)):
            batch['image'] = batch['image'].to(device)
            batch['text'] = batch['text'].squeeze(dim=1).to(device)
            
            with autocast(enabled=use_amp):
                image_features, text_features = model(batch)
                loss_g = clip_loss(image_features, text_features, logit)
                loss = loss_g 
                train_losses.append(loss.item())
                if step % 100 == 0:
                    logger.info(f'{epoch}/{step}: Constrastive loss {loss} Global loss {loss_g}')
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        train_loss = np.mean(train_losses)
        writer.add_scalar('Training loss', train_loss, epoch)
        logger.info(f'epoch:{epoch} training loss is {train_loss}')
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch['image'] = batch['image'].to(device)
                batch['text'] = batch['text'].squeeze(dim=1).to(device)
                with autocast(enabled=use_amp):
                    image_features, text_features = model(batch)
                    loss_g = clip_loss(image_features, text_features, logit)
                    loss = loss_g 
                if step % 10 == 0:
                    logger.info(f'{epoch}/{step}: Constrastive loss {loss} Global loss {loss_g}')
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            writer.add_scalar('Validation loss', val_loss, epoch)
            logger.info(f'epoch:{epoch} validation loss is {val_loss}')

            for batch in tqdm(test_loader):
                batch['image'] = batch['image'].to(device)
                batch['text'] = batch['text'].squeeze(dim=1).to(device)
                with autocast(enabled=use_amp):
                    image_features, text_features = model(batch)
                    loss_g = clip_loss(image_features, text_features, logit)
                    loss = loss_g 
                if step % 10 == 0:
                    logger.info(f'{epoch}/{step}: Constrastive loss {loss} Global loss {loss_g}')
                test_losses.append(loss.item())
            test_loss = np.mean(test_losses)
            writer.add_scalar('Testing loss', test_loss, epoch)
            logger.info(f'epoch:{epoch} testing loss is {test_loss}')

            torch.save(model.state_dict(), os.path.join(log_dir, f'{epoch}-{round(val_loss, 4)}-{round(test_loss, 4)}.pth'))
    logger.info('Training finish...')
            


