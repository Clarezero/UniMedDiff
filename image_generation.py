import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image
import open_clip

from tqdm import tqdm

from bio_con_model import MyCLIPEncoder_squeeze77


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    # accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    # config.mixed_precision = accelerator.mixed_precision
    config.mixed_precision = True
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    with open(config.input_path, 'r') as f:
        prompts = f.read().strip().split('\n')
    


    with open(config.input_cls_path, 'r') as f:
        infos = f.read().strip().split('\n')
    infos = [info.split('&&&')[1].strip().split(' ') for info in infos]


    class_prompts = []
    class_noun = ['There is atelectasis.', 'There is cardiomegaly.', 'There is consolidation.', 'There is edema.', 
                  'There is enlarged cardiomediastinum.', 'There is fracture.', 'There is lung lesion.', 'There is lung opacity.', 
                  'There is pleural effusion.', 'There is pneumonia.', 'There is pneumothorax.']
    # t=0
    # for info in infos:
    #     index = 0
    #     class_prior = None
    #     for step, class_type in enumerate(info):
    #         if step==8:
    #             if class_type == '1':
    #                 t+=1
    # print(t)

    for info in infos:
        index = 0
        class_prior = None
        for step, class_type in enumerate(info):
            if step == 8 or step == 10 or step ==  13:
                continue
            
                if class_prior is None:
                    class_prior = class_noun[index]
                else:
                    class_prior = class_prior + ' ' + class_noun[index]
            index = index + 1            
        if class_prior is None:
            class_prior = 'There is no finding.'
        class_prompts.append(class_prior)
    # clip = libs.clip.FrozenCLIPEmbedder()
    # clip.eval()
    # clip.to(device)

    # contexts = clip.encode(prompts)

    ### clip text encoding
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    clip = MyCLIPEncoder_squeeze77(backbone)
    clip.eval()
    # path = '/storage/ScientificPrograms/Diffusion/code/BioMedClip_finetune/FinalData/Contrastive_Pretrain_v1/temp007/Sep04_08-12-31/15-2.1513-1.9602.pth'
    path = 'weights/Contrastive_Pretrain_v1_CL_77_seed3407/15-2.2943-2.0637.pth'
    clip.load_state_dict(torch.load(path), strict=False)
    clip = clip.to(device)
    ###

    # with torch.no_grad():
    #     contexts = clip(tokenizer(prompts, context_length=256).to(device))
    #     class_contexts = clip(tokenizer(class_prompts, context_length=256).to(device))


    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    # nnet_path = '/storage/ScientificPrograms/Diffusion/code/U-ViT-main/workdir/NLMCXR_256/depth=16/ckpts/300000.ckpt/nnet.pth'
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context, hint):
        _cond = nnet(x, timesteps, context=context, hint=hint)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, dtype=torch.float32, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context, hint=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    # 定义一个函数，用于对输入的_batch进行编码
    # 定义一个函数，用于对输入的_batch进行编码
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config.sample)
    logging.info(f'mixed_precision={config.mixed_precision}')
    logging.info(f'N={N}')

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    batch_size = 4
    ttl_nums = len(prompts)
    chunks = ttl_nums // batch_size
    final_bs = ttl_nums % batch_size
    os.makedirs(config.output_path, exist_ok=True)

    for i in tqdm(range(chunks + 1)):
        if os.path.exists(os.path.join(config.output_path, f"{i}-0.png")):
            continue
        if i != chunks:
            z_init = torch.randn(batch_size, *config.z_shape, device=device)
            # batch_contexts = contexts[i * 4 : (i + 1) * 4, :, :]
            # batch_class_contexts = class_contexts[i * 4 : (i + 1) * 4, :, :]
            with torch.no_grad():
                batch_prompts = prompts[i * 4 : (i + 1) * 4]
                batch_class_prompts = class_prompts[i * 4 : (i + 1) * 4]
                batch_contexts = clip(tokenizer(batch_prompts, context_length=256).to(device))
                batch_class_contexts = clip(tokenizer(batch_class_prompts, context_length=256).to(device))
        else:
            z_init = torch.randn(final_bs, *config.z_shape, device=device)
            # batch_contexts = contexts[i * 4: , :, :]
            # batch_class_contexts = class_contexts[i * 4: , :, :]
            # batch_prompts = prompts[i * 4 : ]
            with torch.no_grad():
                batch_prompts = prompts[i * 4 : ]
                batch_class_prompts = class_prompts[i * 4 : ]
                batch_contexts = clip(tokenizer(batch_prompts, context_length=256).to(device))
                batch_class_contexts = clip(tokenizer(batch_class_prompts, context_length=256).to(device))      

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context=batch_contexts, hint=batch_class_contexts)
        
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        z = dpm_solver.sample(z_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
        samples = dataset.unpreprocess(decode(z))

        for step, (sample, prompt) in enumerate(zip(samples, batch_prompts)):
            save_image(sample, os.path.join(config.output_path, f"{i}-{step}.png"))
            


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output images.")
flags.DEFINE_string("input_path", None, "The path to input texts.")
flags.DEFINE_string("input_cls_path", None, "The path to input texts' classes.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_path = FLAGS.input_path
    config.input_cls_path = FLAGS.input_cls_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
