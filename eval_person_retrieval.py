import argparse
import os
# import ruamel_yaml as yaml
import ruamel.yaml as yaml
import numpy as np
import random
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.blip_retrieval import blip_retrieval
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
import utils
from interactive.vqa import vqa_retrieval
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dataset import create_dataset
from dataset.re_dataset import TextMaskingGenerator
from models.model_retrieval import CIM_Retrieval


os.environ['CUDA_VISIBLE_DEVICES'] = "3"
def main(args, config, config_vqa, config_cap=None):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")
    
    _, _, test_dataset = create_dataset(args.task, config, evaluate=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_test'],
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    #### Model ####
    print("Creating model")

    model = CIM_Retrieval(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=True)
    model = model.to(device)

    model_vqa = blip_vqa(pretrained=config_vqa['pretrained'], image_size=config_vqa['image_size'],
                         vit=config_vqa['vit'], vit_grad_ckpt=config_vqa['vit_grad_ckpt'],
                         vit_ckpt_layer=config_vqa['vit_ckpt_layer'])
    model_vqa = model_vqa.to(device)

    model_cap = None
    if args.automatic:
        model_cap = blip_decoder(pretrained=config_cap['pretrained'], image_size=config_cap['image_size'],
                                 vit=config_cap['vit'], prompt=config_cap['prompt'])
        model_cap = model_cap.to(device)
    eval_results = vqa_retrieval(model_vqa, model, test_loader, config['k_test'], config=config, args=args)
    print(eval_results)
    log_stats = {**{f'{k}': v for k, v in eval_results.items()}, }

    with open(os.path.join(args.output_dir, f"test_results.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqa_config', default='./configs/vqa.yaml')
    parser.add_argument('--cap_config', default='./configs/nocaps.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_msrvtt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--aggregate', action='store_true', default=False)
    parser.add_argument('--automatic', action='store_true', default=False)
    parser.add_argument('--task', default='re_rstp')
    args = parser.parse_args()
    config = yaml.load(open('./configs/retrieval_rstp.yaml', 'r'), Loader=yaml.Loader)
    config_vqa = yaml.load(open(args.vqa_config, 'r'), Loader=yaml.Loader)
    config_cap = yaml.load(open(args.cap_config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config, config_vqa, config_cap)