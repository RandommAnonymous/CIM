import os
import math
import copy
import numpy as np
import time
import datetime
import json
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw
from sklearn import metrics
from easydict import EasyDict
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils

from dataset.re_dataset import TextMaskingGenerator
from train_tools import mlm


@torch.no_grad()
def accs(pred, y):
    print('Testing ... metrics')
    num_persons = pred.shape[0]
    print('num_persons', num_persons)
    ins_acc = 0
    ins_prec = 0
    ins_rec = 0
    mA_history = {
        'correct_pos': 0,
        'real_pos': 0,
        'correct_neg': 0,
        'real_neg': 0
    }

    # compute label-based metric
    outputs = pred
    attrs = y
    overlaps = outputs * attrs
    mA_history['correct_pos'] += overlaps.sum(0)
    mA_history['real_pos'] += attrs.sum(0)
    inv_overlaps = (1 - outputs) * (1 - attrs)
    mA_history['correct_neg'] += inv_overlaps.sum(0)
    mA_history['real_neg'] += (1 - attrs).sum(0)

    outputs = outputs.astype(bool)
    attrs = attrs.astype(bool)

    # compute instabce-based accuracy
    intersect = (outputs & attrs).astype(float)
    union = (outputs | attrs).astype(float)
    ins_acc += (intersect.sum(1) / union.sum(1)).sum()
    ins_prec += (intersect.sum(1) / outputs.astype(float).sum(1)).sum()
    ins_rec += (intersect.sum(1) / attrs.astype(float).sum(1)).sum()

    ins_acc /= num_persons
    ins_prec /= num_persons
    ins_rec /= num_persons
    ins_f1 = (2 * ins_prec * ins_rec) / (ins_prec + ins_rec)

    term1 = mA_history['correct_pos'] / mA_history['real_pos']
    term2 = mA_history['correct_neg'] / mA_history['real_neg']
    label_mA_verbose = (term1 + term2) * 0.5
    label_mA = label_mA_verbose.mean()

    print('* Results *')
    print('  # test persons: {}'.format(num_persons))
    print('  (label-based)     mean accuracy: {:.2%}'.format(label_mA))
    print('  (instance-based)  accuracy:      {:.2%}'.format(ins_acc))
    print('  (instance-based)  precition:     {:.2%}'.format(ins_prec))
    print('  (instance-based)  recall:        {:.2%}'.format(ins_rec))
    print('  (instance-based)  f1-score:      {:.2%}'.format(ins_f1))
    print('  mA for each attribute: {}'.format(label_mA_verbose))
    return label_mA, ins_acc, ins_prec, ins_rec, ins_f1


@torch.no_grad()
def itm_eval_attr(scores_i2t, dataset):
    label = dataset.label
    pred = []
    for i in range(label.shape[1]):
        a = np.argmax(scores_i2t[:, 2 * i: 2 * i + 2], axis=1)
        pred.append(a)

    label_mA, ins_acc, ins_prec, ins_rec, ins_f1 = accs(np.array(pred).T, label)
    print('############################################################\n')
    eval_result = {'label_mA': round(label_mA, 4),
                   'ins_acc': round(ins_acc, 4),
                   'ins_prec': round(ins_prec, 4),
                   'ins_rec': round(ins_rec, 4),
                   'ins_f1': round(ins_f1, 4),
                   }
    return eval_result


@torch.no_grad()
def itm_eval_attr_only_img_classifier(scores_i2t, dataset):
    label = dataset.label
    pred = scores_i2t
    label_mA, ins_acc, ins_prec, ins_rec, ins_f1 = accs(pred, label)
    print('############################################################\n')
    eval_result = {'label_mA': round(label_mA, 4),
                   'ins_acc': round(ins_acc, 4),
                   'ins_prec': round(ins_prec, 4),
                   'ins_rec': round(ins_rec, 4),
                   'ins_f1': round(ins_f1, 4),
                   }
    return eval_result


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args, diffusion = None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()
    list_g_ids = [int(num) for num in data_loader.dataset.g_pids]
    list_g_ids = torch.tensor(list_g_ids)
    print(list_g_ids.shape)
    # print(list_g_ids)
    g_ids = []
    texts = data_loader.dataset.text
    gpts = data_loader.dataset.gpt
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

def mAP(scores_t2i, g_pids, q_pids, dataloader,table=None):
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
    diff = 2 * cal_diff(matches) / matches.shape[0]
    all_cmc = matches[:, :10].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    if not table:
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP
                   }
    return eval_result
