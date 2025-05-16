# 自动
conda create -n diff_cxr python=3.12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e ./repos/open_clip -e ./repos/timm
pip install einops requests tqdm pyyaml packaging regex opencv-python tensorboard ml-collections accelerate wandb spicy ftfy

# 手动--------------------------------------------------------------------------------------------
conda create -n diff_cxr python=3.12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install einops
pip install requests
pip install tqdm
pip install pyyaml
pip install packaging
pip install regex
pip install opencv-python
pip install open-clip-torch==2.20.0
pip install tensorboard
pip install ml-collections
pip install accelerate
pip install wandb
pip install spicy 

#open_clip修改
C:\Users\Dell\.conda\envs\diff_cxr\Lib\site-packages\open_clip\model.py
class CLIPTextCfg:
    pooler_type: str = 'cls_last_hidden_state_pooler'
    output_tokens: bool = True
    hf_proj_type: str = None 
    hf_pooler_type: str = None

C:\Users\Dell\.conda\envs\diff_cxr\Lib\site-packages\open_clip\factory.py:
def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    del state_dict['text.transformer.embeddings.position_ids']
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

C:\Users\Dell\.conda\envs\diff_cxr\Lib\site-packages\open_clip\hf_model.py
if self.output_tokens:
    return out.last_hidden_state[:, self.pooler.cls_token_position, :], tokens

# C:\Users\Dell\.cache\huggingface\hub\models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224\snapshots\9f341de24bfb00180f1b847274256e9b65a3a32e\open_clip_config.json
# open_clip_config Clspooler替换mean pooler

C:\Users\Dell\.conda\envs\diff_cxr\Lib\site-packages\open_clip\timm_model.py
token_0, x = self.trunk(x)
# x = self.head(x)
return token_0, x


# timm修改
C:\Users\Dell\.conda\envs\diff_cxr\Lib\site-packages\timm\models\vision_transformer.py
class VisionTransformer(nn.Module):
def forward_head(self, x, pre_logits: bool = False):
    latent = x[:, 1:, :]
    if self.global_pool:
        x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    token_0 = x
    # x = self.fc_norm(x)
    # x = self.head_drop(x)
    # return (x, token_0) if pre_logits else (self.head(x), token_0)
    # return (x, token_0) if pre_logits else (x[:, 1:, :], token_0)
    return latent, token_0
def forward(self, x):
    features = self.forward_features(x)
    # features: torch.Size([20, 197, 768])
    x, token_0 = self.forward_head(features)
    # torch.Size([20, 196, 768]) torch.Size([20, 768])        
    return token_0, x


