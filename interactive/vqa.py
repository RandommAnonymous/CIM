from interactive.extract_feature import *
from interactive.gen_cap import *
import torch.distributed as dist
from reTools import mAP


import numpy as np
import time
import datetime

from easydict import EasyDict
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils

from dataset.re_dataset import TextMaskingGenerator
from train_tools import mlm
@torch.no_grad()
def vqa_retrieval(model_vqa, model_retrieval, data_loader, k_test, config=None, args=None):

    device = torch.device('cuda')

    images, image_ids, image_embeds, image_feats = get_image_embeds(model_retrieval, data_loader, device=device)
    # del image_ids, image_embeds
    # images = data_loader.dataset.image
    tokenizer = model_retrieval.tokenizer
    num_images = len(images)

    total_caption = []
    if True:
        for i in range(num_images):


            image = images[i].to(device, non_blocking=True).unsqueeze(0)

            captions = []
            answer = generate_caption_vqa(image, model_vqa)

            for v in answer:
                captions.append(v)
            print(captions)
            total_caption.append(captions)
            total_caption.append(captions)
        total_caption = list(zip(*total_caption))

    dialog_sims = []
    for one_caption in total_caption:
        one_feats = []
        for i in range(0, len(one_caption), config['batch_size_test_text']):
            text = one_caption[i: min(len(one_caption), i + config['batch_size_test_text'])]
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                return_tensors="pt").to(device)
            text_embed = model_retrieval.get_text_embeds(text_input.input_ids, text_input.attention_mask)
            text_feat = model_retrieval.text_proj(text_embed[:, 0, :])
            text_feat = F.normalize(text_feat, dim=-1)
            one_feats.append(text_feat)
        one_feats = torch.cat(one_feats, dim=0)
        sims = one_feats @ image_feats.t()
        dialog_sims.append(sims)

    dialog_sims = torch.vstack(dialog_sims)
    dialog_sims = torch.mean(dialog_sims, dim=0, keepdim=True)

    score_matrix_t2i, _ = evaluation(model_retrieval, data_loader, dialog_sims, tokenizer, device, config, args)


    eval_result = mAP(score_matrix_t2i, data_loader.dataset.g_pids, data_loader.dataset.q_pids, None,table=None)

    return eval_result


@torch.no_grad()
def evaluation(model, data_loader, dialog_sims, tokenizer, device, config, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    texts = data_loader.dataset.text
    gpts = data_loader.dataset.gpt
    print('Computing features for evaluation...')
    start_time = time.time()

    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_embeds = []
    text_atts = []
    text_feats = []
    gpt_embeds = []
    gpt_atts = []
    gpt_feats = []
    text_feats_masked = []
    mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                        config['skipgram_prb'], config['skipgram_size'],
                                        config['mask_whole_word'])
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.text_proj(text_embed[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        # mlm start
        text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                    config)
        text_embed_masked = model.get_text_embeds(text_ids_masked, text_input.attention_mask)
        text_feat_masked = model.text_proj(text_embed_masked[:, 0, :])
        text_feat_masked = F.normalize(text_feat, dim=-1)
        text_feats_masked.append(text_feat_masked)
        # mlm end

        text_embeds.append(text_embed)
        text_atts.append(text_input.attention_mask)
        text_feats.append(text_feat)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_feats_masked = torch.cat(text_feats_masked, dim=0)

    image_embeds = []
    image_feats = []
    image_ids = []
    image_atts = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed, image_att = model.get_vision_embeds(image)
        image_feat = model.vision_proj(image_embed[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)
        image_atts.append(image_att)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
        image_ids.append(img_id)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_atts = torch.cat(image_atts, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    image_ids = torch.cat(image_ids, dim=0)
    if config['gpt']:
        for i in range(0, len(gpts), text_bs):
            gpt = gpts[i:min(len(gpts), i + text_bs)]
            gpt_input = tokenizer(gpt, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                return_tensors="pt").to(device)
            gpt_embed = model.get_text_embeds(gpt_input.input_ids, gpt_input.attention_mask)
            gpt_feat = model.text_proj(gpt_embed[:, 0, :])
            gpt_feat = F.normalize(gpt_feat, dim=-1)

            gpt_embeds.append(gpt_embed)
            gpt_atts.append(gpt_input.attention_mask)
            gpt_feats.append(gpt_feat)
        gpt_embeds = torch.cat(gpt_embeds, dim=0)
        gpt_atts = torch.cat(gpt_atts, dim=0)
        gpt_feats = torch.cat(gpt_feats, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    sims_matrix = sims_matrix.t()
    if config['gpt']:
        gpt_img_sim = gpt_feats @ image_feats.t()
        sims_matrix = (sims_matrix + gpt_img_sim) / 2

    sims_matrix = sims_matrix + dialog_sims
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -1000.0).to(device)
    score_sim_t2i = sims_matrix
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.get_cross_embeds(encoder_output, encoder_att,
                                        text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1))[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
        score_sim_t2i[start + i, topk_idx] = topk_sim
    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i



    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    per_time = total_time / num_text
    print('total_time', total_time)
    print('per_time', per_time)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    masked_matrix =  text_feats_masked @ image_feats.t()
    return score_matrix_t2i.cpu().numpy(), masked_matrix
