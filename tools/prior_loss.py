from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        # for vision input [B, I]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)


        # for text input [B, T]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 =  self.value2(input_tensor2)

        
        attention_scores1  = query_layer2 @ key_layer1.T # [T, D] @ [D, I] = [T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
     
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 =  attention_probs1 @ value_layer1 # [T, I] @ [I, D] = [T, D]
        attention_scores2 = query_layer1 @ key_layer2.T # [I, D] @ [D, T] = [I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = attention_probs2 @ value_layer2 # [I, T] @ [T, D] = [I, D]
        return context_layer2, attention_probs2, context_layer1, attention_probs1


def text_local_loss_fn(embed_A, embed_B, norm=True):
    '''
    Similarly to CUT[1], we only utilized internal negative samples in a single report. 
    Although incorporating additional negative sentences from other patients could potentially provide more negative samples, we observed a decline in performance. This outcome is understandable, as different reports may contain highly similar sentences (especially for normal sample).
    [1] Park T, Efros A A, Zhang R, et al. Contrastive learning for unpaired image-to-image translation[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16. Springer International Publishing, 2020: 319-345.
    '''
    local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = local_logit_scale.exp()
    if norm:
        embed_A = F.normalize(embed_A, dim=-1, p=2)
        embed_B = F.normalize(embed_B, dim=-1, p=2)
    lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
    logits_per_image = logit_scale * embed_B @ embed_A.t()
    logits_per_text = logit_scale * embed_A @ embed_B.t()
    image_loss = F.cross_entropy(logits_per_image, lc_labels)
    text_loss = F.cross_entropy(logits_per_text, lc_labels)
    loss = (image_loss + text_loss) / 2   
    return loss
   

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