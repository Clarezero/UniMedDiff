import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange, reduce


def masked_mean(t, mask, dim=1, eps=1e-6):
    """
    Compute the mean of tensor `t` along dimension `dim` using only the masked elements.
    """
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom

def max_neg_value(dtype):
    """
    Return the most negative representable value for the given dtype.
    """
    return -torch.finfo(dtype).max

def matrix_diag(t):
    """
    Extract the diagonal elements from the last two dimensions of a 2D or 3D tensor.
    """
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)

def log(t, eps=1e-20):
    """
    Safe logarithm: log(t + eps) to avoid log(0).
    """
    return torch.log(t + eps)



class MyCLIPEncoder(nn.Module):
    """
    Basic CLIP text encoder wrapper.
    """
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

    def forward(self, token_embeddings):
        _, text_features = self.text(token_embeddings)
        return text_features


class MyCLIP(nn.Module):
    """
    CLIP model wrapper that returns both image and text features.
    """
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIP, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

    def forward(self, batch):
        image_features, _ = self.visual(batch['image'])
        text_features, _ = self.text(batch['text'])
        return image_features, text_features


class MyCLIPEncoder_squeeze77(nn.Module):
    """
    CLIP text encoder with a 1D Conv-based feature squeeze module (to 77 tokens).
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
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


class MyCLIPEncoder_squeeze32(nn.Module):
    """
    CLIP text encoder with a squeeze module (to 32 tokens).
    """
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze32, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        
    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


class MyCLIPEncoder_squeeze128(nn.Module):
    """
    CLIP text encoder with a squeeze module (to 128 tokens).
    """
    def __init__(self, backbone, text_pad_id=0):
        super(MyCLIPEncoder_squeeze128, self).__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_pad_id = text_pad_id

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 129, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        
    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features

    
def simsiam_loss_func(x, y, predictor):
    """
    Compute SimSiam loss between two feature sets x and y using the predictor network.
    """
    p_x = predictor(x)
    p_y = predictor(y)
    z_x = x.detach()
    z_y = y.detach()
    return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() +
              F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5



def simsiam_loss(image_to_local_text_embed, text_to_local_image_embed, local_image_embed, local_text_embed): 
    """
    SimSiam loss between image-to-text and text-to-image local embeddings.
    """
    predictor = nn.Sequential(
        nn.Linear(768, 384),
        nn.ReLU(inplace=True),
        nn.Linear(384, 768)
    )
    text_loss = simsiam_loss_func(image_to_local_text_embed, local_text_embed, predictor)
    image_loss = simsiam_loss_func(text_to_local_image_embed, local_image_embed, predictor)
    return image_loss, text_loss


class MyCLIPSqueeze_GLv1(nn.Module):
    """
    CLIP text encoder with an information squeezing block and local attention layers.
    Used to compress text features into fewer patches.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        
        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 768)
        )

        self.patch_local_atten_layer = nn.MultiheadAttention(768, 1)
        self.word_local_atten_layer = nn.MultiheadAttention(768, 1)

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)

        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)

        squeeze_text = self.info_squeeze(text_info)
        text_patches = squeeze_text[:, 1:, :]
        return text_patches



class MyCLIPSqueezeEncoder_Attn1(nn.Module):
    """
    Variant with an attention block before info squeezing.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.text_pad_id = text_pad_id

        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.info_squeeze = nn.Sequential(
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features



class MyCLIPEncoderSqueeze_GLv1(nn.Module):
    """
    Another variant with the same structure as GLv1.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 768)
        )

        self.patch_local_atten_layer = nn.MultiheadAttention(768, 1)
        self.word_local_atten_layer = nn.MultiheadAttention(768, 1)

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_patches = squeeze_text[:, 1:, :]
        return text_patches



class MyCLIPSqueeze_GLv1_32(nn.Module):
    """
    Same as GLv1 but squeezes to 32 patches instead of 78.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 768)
        )

        self.patch_local_atten_layer = nn.MultiheadAttention(768, 1)
        self.word_local_atten_layer = nn.MultiheadAttention(768, 1)

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_patches = squeeze_text[:, 1:, :]
        return text_patches


class MyCLIPEncoderSqueeze_prior(nn.Module):
    """
    Text encoder with learnable prototypes (prior knowledge) and a class prediction head.
    Also performs info squeezing and local alignment.
    """
    def __init__(self, backbone, num_prototypes):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual

        self.info_squeeze = nn.Sequential(
            nn.Conv1d(257, 257, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 78, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(768, 768 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(768 // 2, 768)
        )

        self.prototype = nn.Embedding(num_prototypes, 768)
        self.patch_local_atten_layer = nn.MultiheadAttention(768, 1)
        self.word_local_atten_layer = nn.MultiheadAttention(768, 1)

        self.extract_global_image_prior = nn.MultiheadAttention(768, 1, batch_first=True)

        self.class_layer = nn.Linear(768, 12)

    def forward(self, token_embeddings):
        b = token_embeddings.shape[0]

        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        squeeze_text = self.info_squeeze(text_info)
        text_patches = squeeze_text[:, 1:, :]

        text_flatten = text_patches.reshape(-1, 768)
        dist = text_flatten @ self.prototype.weight.T
        soft_one_hot = F.gumbel_softmax(dist, tau=0.9, dim=1, hard=False)
        output = soft_one_hot @ self.prototype.weight

        image_prior = output.reshape(b, 77, -1)
        image_prior, _ = self.extract_global_image_prior(image_prior, image_prior, image_prior)
        image_prior = image_prior[:, 0, :]

        class_logits = self.class_layer(image_prior)

        return text_patches, image_prior




class NaturalCLIPEncoder(nn.Module):
    """
    Simple wrapper for HuggingFace-like text encoder to output last hidden states.
    """
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, token_embeddings):
        text_outputs = self.text_model(input_ids=token_embeddings)
        return text_outputs.last_hidden_state


class MyCLIPSqueezeEncoder_Attn32(nn.Module):
    """
    Attention + squeeze version, squeezing to 32 patches.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.text_pad_id = text_pad_id

        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.info_squeeze = nn.Sequential(
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 33, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
        text_info, _ = self.info_Attn(text_info, text_info, text_info)
        squeeze_text = self.info_squeeze(text_info)
        text_features = squeeze_text[:, 1:, :]
        return text_features


class MyCLIPSqueezeEncoder_Attn128(nn.Module):
    """
    Attention + squeeze version, squeezing to 128 patches.
    """
    def __init__(self, backbone, text_pad_id=0):
        super().__init__()
        self.text = backbone.text
        self.visual = backbone.visual
        self.text_pad_id = text_pad_id

        self.info_Attn = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.info_squeeze = nn.Sequential(
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Conv1d(257, 129, 3, padding=1),
            nn.LayerNorm(768),
            nn.ReLU()
        )

    def forward(self, token_embeddings):
        text_features, text_latent = self.text(token_embeddings)
        text_info = torch.cat((text_features.unsqueeze(1), text_latent), dim=1)
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
