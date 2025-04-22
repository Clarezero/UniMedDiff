import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F


from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


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
        # [1, 512] [1, 256, 768]

        return text_features


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
        #torch.Size([4, 512]) torch.Size([4, 256, 768])
        #([4, 768]) torch.Size([4, 256, 768])
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]

        return text_features


class MyCLIPEncoder_squeeze32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze32, self).__init__()
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
        
    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        # image_features, _= self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]

        return text_features


class MyCLIPEncoder_squeeze128(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze128, self).__init__()
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
        
    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        # image_features, _= self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]

        return text_features
    

### 
def simsiam_loss_func(x, y, predictor, flag='image'):
    p_x = predictor(x)
    p_y = predictor(y)
    z_x = x.detach()
    z_y = y.detach()
    return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5


def simsiam_loss(image_to_local_text_embed, text_to_local_image_embed, local_image_embed, local_text_embed): 
    '''
    The convolutions in encoder may cause overlap between the receptive fields of the patches, a simple negative sampling strategy is not applicable.
    '''
    predictor = nn.Sequential(nn.Linear(768, 768 // 2),
                                    nn.ReLU(inplace=True), # hidden layer 
                                    nn.Linear(768 // 2, 768)) # output layer # used for simsiam loss
    text_loss = simsiam_loss_func(image_to_local_text_embed, local_text_embed, predictor, flag='text')
    image_loss = simsiam_loss_func(text_to_local_image_embed, local_image_embed, predictor, flag='image')
    return  image_loss, text_loss 

class MyCLIPSqueeze_GLv1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_GLv1, self).__init__()
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

        self.predictor = nn.Sequential(nn.Linear(768, 768 // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(768 // 2, 768)) # output layer # used for simsiam loss
        

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        # token local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1)

    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        # _, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        ### global features and info squeezes
        ### 对于压缩后的文本添加一个local contrastive loss
        
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]
        
            # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        return text_patches


class MyCLIPSqueezeEncoder_Attn1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueezeEncoder_Attn1, self).__init__()
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

    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        # image_features, _ = self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


class MyCLIPEncoderSqueeze_GLv1(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoderSqueeze_GLv1, self).__init__()
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

        self.predictor = nn.Sequential(nn.Linear(768, 768 // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(768 // 2, 768)) # output layer # used for simsiam loss
        

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        # token local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1)

    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        # image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        ### global features and info squeezes
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]

        # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        return text_patches



class MyCLIPSqueeze_GLv1_32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueeze_GLv1_32, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
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

    def forward(self, token_embeddings):
        # b, device = batch['text'].shape[0], batch['text'].device
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        # image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        ### global features and info squeezes
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        # text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]

        # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        return text_patches
    

class MyCLIPEncoderSqueeze_prior(nn.Module):
    def __init__(self, backbone, num_prototypes):
        super(MyCLIPEncoderSqueeze_prior, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        # self.clip = ClipLoss()

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
        

        self.prototype = nn.Embedding(num_prototypes, 768)
        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        # token local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            768, 1)
        ### image prior
        self.extract_global_image_prior = nn.MultiheadAttention(
            768, 1, batch_first=True)
        self.class_layer = nn.Linear(768, 12)

    def forward(self, token_embeddings):
        b = token_embeddings.shape[0]
        ### features obtain
        ### img features B 96 768
        ### text features B 77 768
        # image_features, image_patches = self.visual(batch['image'])
        text_features, text_latent = self.text(token_embeddings)
        ### global features and info squeezes
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 0, :]
        text_patches = squeeze_text[:, 1:, :]
        
        ### local feature alignment
        ### calculation
        # total_image_loss = 0
        # total_text_loss = 0
        # for idx in range(b):
        #     local_text_embed = text_patches[idx] # get sentence-level representation 
        #     local_image_embed = image_patches[idx] # get patch-level representation
            
        #     text_to_local_image_embed, text_to_local_image_atten = self.patch_local_atten_layer(local_image_embed, local_text_embed, local_text_embed)
        #     image_to_local_text_embed, image_to_local_text_atten = self.word_local_atten_layer(local_text_embed, local_image_embed, local_image_embed) 
            
        #     text_to_local_image_embed = F.normalize(text_to_local_image_embed, dim=-1)
        #     image_to_local_text_embed = F.normalize(image_to_local_text_embed, dim=-1)
            
        #     # for local text-to-image alignment, we employ the simsiam loss without negative sample 
        #     image_loss = simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
        #     # for local image-to-text alignment, we just use the contrastive loss
        #     # text_loss = text_local_loss_fn(local_text_embed, image_to_local_text_embed)
        #     text_loss = self.clip(local_text_embed, image_to_local_text_embed, torch.tensor(1 / 0.10).to(device))
        #     total_image_loss += image_loss
        #     total_text_loss += text_loss

            # text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        
        ### medical prior info (image distribution and classes info)
        text_flatten = text_patches.reshape(-1, 768)
        dist = text_flatten @ self.prototype.weight.T
        self.gt_dist = dist
        soft_one_hot = F.gumbel_softmax(dist, tau=0.9, dim=1, hard=False)
        output = soft_one_hot @ self.prototype.weight

        ### explicitly recon the text embedding
        # recon_loss = (output - text_flatten).abs().mean()

        ### global image alignment (cos similarity or contrastive loss)
        image_prior = output.reshape(b, 77, -1)
        image_prior, _ = self.extract_global_image_prior(image_prior, image_prior, image_prior)
        image_prior = image_prior[:, 0, :]

        ### class info
        class_logits = self.class_layer(image_prior)

        return text_patches, image_prior


class NaturalCLIPEncoder(nn.Module):
    def __init__(self, text_model):
        super(NaturalCLIPEncoder, self).__init__()
        self.text_model = text_model
        # self.vision_model = vision_model
        # self.visual_projection  = nn.Linear(1024, 768)

    def forward(self, token_embeddings):
        # b, device = batch['image'].shape[0], batch['image'].device
        ### features obtain
        # image_ouputs = self.vision_model(batch['image'])
        # projection = self.visual_projection(torch.cat((image_ouputs.pooler_output.unsqueeze(1), image_ouputs.last_hidden_state), axis=1))
        text_outputs= self.text_model(input_ids=token_embeddings)
        return text_outputs.last_hidden_state


class MyCLIPSqueezeEncoder_Attn32(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueezeEncoder_Attn32, self).__init__()
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

    def forward(self, token_embeddings):
        ### features obtain
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


class MyCLIPSqueezeEncoder_Attn128(nn.Module):
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPSqueezeEncoder_Attn128, self).__init__()
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

    def forward(self, token_embeddings):
        ### features obtain
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(dim=1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


if __name__ == '__main__':
    import open_clip
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = MyCLIP(backbone)

    text = torch.randint(0, 30522, (4, 256))
    images = torch.randn(4, 3, 224, 224)

    loss = model({'text':text, 'image':images})
    print(loss)