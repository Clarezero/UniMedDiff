import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from tools.prior_loss import simsiam_loss_func, text_local_loss_fn
from tools.clip_loss import ClipLoss

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

def log(t, eps = 1e-20):
    return torch.log(t + eps)

class MyCLIPEncoder(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        # image_features, _= self.visual(batch['image'])
        _, text_features = self.text(token_embeddings)

        return text_features


class MyCLIPEncoder_squeeze77(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze77, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )
    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        # image_features, _= self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]

        return text_features



class MyCLIPCls_squeeze77(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPCls_squeeze77, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 11),
            nn.ReLU()
        )
    def forward(self, batch):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_feature, _= self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 1:, :]
        text_feature = squeeze_text[:, 0, :]

        cls_out = self.cls_head(text_feature)
        return image_feature, text_feature, cls_out

class MyCLIPCls_squeeze77_ISIC(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPCls_squeeze77_ISIC, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 7),
            nn.ReLU()
        )
    def forward(self, batch):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_feature, _= self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 1:, :]
        text_feature = squeeze_text[:, 0, :]

        cls_out = self.cls_head(text_feature)
        return image_feature, text_feature, cls_out
    


class MyCLIPCls_squeeze77_Colorectal(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPCls_squeeze77_Colorectal, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 3),
            nn.ReLU()
        )
    def forward(self, batch):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_feature, _= self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 1:, :]
        text_feature = squeeze_text[:, 0, :]

        cls_out = self.cls_head(text_feature)
        return image_feature, text_feature, cls_out
    

class MyCLIP(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIP, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, _ = self.text(batch['text'])
        return image_features, text_features


class NaturalCLIP(nn.Module):
    def __init__(self, vision_model, text_model):
        super(NaturalCLIP, self).__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.visual_projection  = nn.Linear(1024, 768)

    def forward(self, batch):
        b, device = batch['image'].shape[0], batch['image'].device
        ### features obtain
        image_ouputs = self.vision_model(batch['image'])
        projection = self.visual_projection(torch.cat((image_ouputs.pooler_output.unsqueeze(1), image_ouputs.last_hidden_state), axis=1))
        text_outputs= self.text_model(input_ids=batch['text'].squeeze(dim=1))
        return projection[:, 0, :], text_outputs.pooler_output


class NaturalCLIPv1(nn.Module):
    def __init__(self, vision_model, text_model):
        super(NaturalCLIP, self).__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.text_projection  = nn.Linear(768, 1024)

    def forward(self, batch):
        b, device = batch['image'].shape[0], batch['image'].device
        ### features obtain
        image_outputs = self.vision_model(batch['image'])
        text_outputs= self.text_model(input_ids=batch['text'].squeeze(dim=1))
        projection = self.text_projection(torch.cat((text_outputs.pooler_output.unsqueeze(1), text_outputs.last_hidden_state), axis=1))
        return image_outputs.pooler_output, projection[:, 0, :]


class MyCLIPSqueeze_v1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_v1, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain

        image_features, _ = self.visual(batch['image'])

        # print(self.visual)
        # image_features = self.visual(batch['image'])

        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_128(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_128, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 129, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features
    
    
class MyCLIPSqueeze_32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_32, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features
    

class MyCLIPSqueeze_Attn(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_Attn, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        
        self.info_squeeze = nn.Sequential(
            # nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_Attn1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_Attn1, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
        self.info_squeeze = nn.Sequential(
            # nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_Attn2(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_Attn2, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        
        self.info_squeeze = nn.Sequential(
            # nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )


    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_Attn32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_Attn32, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
        self.info_squeeze = nn.Sequential(
            # nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_Attn128(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_Attn128, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        
        self.info_squeeze = nn.Sequential(
            # nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 129, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features
    

class MyCLIPSqueeze_GL(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_GL, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual

        ### info squeeze
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        ###
        self.img_squeeze = nn.Sequential(
            nn.Conv1d(196, 196, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(196, 1, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.text_squeeze = nn.Sequential(
            nn.Conv1d(77, 77, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(77, 1, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )
    

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1, batch_first=True)

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        ### global features and info squeezes
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]

        ###
        word_atten_output, _ = self.word_local_atten_layer(
                text_patches, image_patches, image_patches)
        patch_atten_output, _ = self.patch_local_atten_layer(
                image_patches, text_patches, text_patches)
        
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        patch_atten_output = F.normalize(patch_atten_output, dim=-1)

        local_image_feature = self.text_squeeze(word_atten_output).squeeze(dim=1)
        local_word_feature = self.img_squeeze(patch_atten_output).squeeze(dim=1)
        
        return image_features, text_features, local_image_feature, local_word_feature


class MyCLIPSqueeze_GLv1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_GLv1, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.clip = ClipLoss()
        ### info squeeze
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.predictor = nn.Sequential(nn.Linear(768, 768 // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(768 // 2, 768)) # output layer # used for simsiam loss
        

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        # token local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1)

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        ### global features and info squeezes
        ### 对于压缩后的文本添加一个local contrastive loss
        
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]
        
        ### local feature alignment
        ### calculation
        total_image_loss = 0
        total_text_loss = 0
        for idx in range(b):
            local_text_embed = text_patches[idx] # get sentence-level representation 
            local_image_embed = image_patches[idx] # get patch-level representation
            
            text_to_local_image_embed, text_to_local_image_atten = self.patch_local_atten_layer(local_image_embed, local_text_embed, local_text_embed)
            image_to_local_text_embed, image_to_local_text_atten = self.word_local_atten_layer(local_text_embed, local_image_embed, local_image_embed) 
            
            text_to_local_image_embed = F.normalize(text_to_local_image_embed, dim=-1)
            image_to_local_text_embed = F.normalize(image_to_local_text_embed, dim=-1)
            
            # for local text-to-image alignment, we employ the simsiam loss without negative sample 
            image_loss = simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
            # for local image-to-text alignment, we just use the contrastive loss
            # text_loss = text_local_loss_fn(local_text_embed, image_to_local_text_embed)
            # temp 0.10
            # text_loss = self.clip(local_text_embed, image_to_local_text_embed, torch.tensor(1 / 0.10).to(device))
            # temp 0.07
            text_loss = self.clip(local_text_embed, image_to_local_text_embed, torch.tensor(1 / 0.07).to(device))
            total_image_loss += image_loss
            total_text_loss += text_loss

            # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        return image_features, text_features, total_image_loss / b, total_text_loss / b


class MyCLIPSqueeze_GLv1_32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_GLv1_32, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.clip = ClipLoss()
        ### info squeeze
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

        self.predictor = nn.Sequential(nn.Linear(768, 768 // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(768 // 2, 768)) # output layer # used for simsiam loss
        

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        # token local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1)

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        ### global features and info squeezes
        ### 对于压缩后的文本添加一个local contrastive loss
        
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]
        
        ### local feature alignment
        ### calculation
        total_image_loss = 0
        total_text_loss = 0
        for idx in range(b):
            local_text_embed = text_patches[idx] # get sentence-level representation 
            local_image_embed = image_patches[idx] # get patch-level representation
            
            text_to_local_image_embed, text_to_local_image_atten = self.patch_local_atten_layer(local_image_embed, local_text_embed, local_text_embed)
            image_to_local_text_embed, image_to_local_text_atten = self.word_local_atten_layer(local_text_embed, local_image_embed, local_image_embed) 
            
            text_to_local_image_embed = F.normalize(text_to_local_image_embed, dim=-1)
            image_to_local_text_embed = F.normalize(image_to_local_text_embed, dim=-1)
            
            # for local text-to-image alignment, we employ the simsiam loss without negative sample 
            image_loss = simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
            # for local image-to-text alignment, we just use the contrastive loss
            # text_loss = text_local_loss_fn(local_text_embed, image_to_local_text_embed)
            # temp 0.10
            # text_loss = self.clip(local_text_embed, image_to_local_text_embed, torch.tensor(1 / 0.10).to(device))
            # temp 0.07
            text_loss = self.clip(local_text_embed, image_to_local_text_embed, torch.tensor(1 / 0.07).to(device))
            total_image_loss += image_loss
            total_text_loss += text_loss

            # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        return image_features, text_features, total_image_loss / b, total_text_loss / b


class MyCLIPSqueeze_v2(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_v2, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 128, 1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),

            # nn.Conv1d(48, 48, 3, padding=1),
            # nn.LayerNorm(768),
            # nn.ReLU(),

            nn.Conv1d(128, 78, 1),
            nn.LayerNorm(768),
            nn.ReLU(),
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_v3(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_v3, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.conv1 = nn.Sequential(
            nn.Conv1d(257, 257, 1),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(257, 257, 5, padding=2),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257 * 3, 78, 3, padding=1),
            nn.LayerNorm(768),
        )
        

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        bran_1 = self.conv1(text_info)
        bran_2 = self.conv2(text_info)
        bran_3 = self.conv3(text_info)
        bran = torch.cat((bran_1, bran_2, bran_3), dim=1)

        squeeze_text = self.info_squeeze(bran)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIPSqueeze_v4(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_v4, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Conv1d(257, 129, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
            # nn.SiLU(),
        )

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(batch['text'])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        return image_features, text_features


class MyCLIP_attention(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIP_attention, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        # self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

    def forward(self, batch):
        b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        image_features, _= self.visual(batch['image'])
        text_features, _ = self.text(batch['text'])

        # image_features = image_features[:, 1:, :]
        # num_batch_texts = num_batch_images = 1
        # temp = self.temperature.exp()
        # text_mask = batch['text'] != self.text_pad_id

        # text_features = rearrange(text_features, '(m b) ... -> m b ...', m = num_batch_texts)
        # image_features = rearrange(image_features, '(m b) ... -> m b ...', m = num_batch_images)

        # sim_text_to_image = einsum('m x t d, n y i d -> m n x y t i', text_features, image_features) * temp
        # sim_image_to_text = sim_text_to_image

        # text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
        # text_to_image_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t', m = num_batch_texts)
        # text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

        # image_to_text_mask = rearrange(text_mask, '(m b) t -> m 1 b 1 t 1', m = num_batch_texts)
        # masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
        # image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

        # # calculate loss

        # text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        # image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # # exponentiate

        # text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # # numerators

        # text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        # pos_mask = torch.eye(b, device = device, dtype = torch.bool)
        # text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))

        # text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # # loss

        # text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
        # image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

        # # calculate CL loss

        # cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        return image_features, text_features
    

if __name__ == '__main__':
    import open_clip
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = MyCLIP(backbone)

    text = torch.randint(0, 30522, (4, 256))
    images = torch.randn(4, 3, 224, 224)

    loss = model({'text':text, 'image':images})
    print(loss)