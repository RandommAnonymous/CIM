import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import utils
import pdb
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_disc_feats(top_k_image_embeds, top_k_inds, image_feats, n_cluster=3):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(top_k_image_embeds)
    closest_1, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, top_k_image_embeds)
    index = top_k_inds[closest_1]
    return image_feats[index].cuda()

@torch.no_grad()
def get_image_embeds(model, data_loader, device='cuda'):
    image_feats = []
    image_embeds = []
    images = []
    image_ids = []
    trans = transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC)
    for image, image_id in tqdm(data_loader):
        image = image.to(device)
        # B, C, W, H = image.shape()
        # print(image.shape)
        image_embed, image_att = model.get_vision_embeds(image)
        image_feat = model.get_features(image_embed)
        image_feat = F.normalize(image_feat, dim=-1)
        image = trans(image)
    #     image_feat = model.vision_encoder(image)
    #     image_embed = model.vision_proj(image_feat[:,0,:])
    #     image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
        images.append(image)
        image_ids.extend(image_id)
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    images = torch.cat(images, dim=0)
    return images, image_ids, image_embeds, image_feats


def get_single_text_embed(model, caption, device='cuda'):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
    return text_embed

def get_text_embeds(model, texts, text_bs=256, device='cuda'):
    tokenizer = model.tokenizer
    num_text = len(texts)
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]
    return text_embeds, text_ids, text_atts

def get_i2t_score_matrix(sims_matrix, image_feats, text_ids, text_atts, model, k_test, device='cuda'):
    score_matrix_i2t = torch.full((len(sims_matrix), sims_matrix.shape[1]), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(device, non_blocking=True)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim
    return score_matrix_i2t

def get_t2i_score_matrix(sims_matrix, image_feats, text_ids, text_atts, model, k_test, device='cuda'):
    score_matrix_t2i = torch.full((len(sims_matrix),sims_matrix.shape[1]),-100.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    for i,sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        topk_idx = topk_idx.cpu()
        encoder_output = image_feats[topk_idx].to(device,non_blocking=True)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True)
        output = model.text_encoder(text_ids[start+i].repeat(k_test,1),
                                    attention_mask = text_atts[start+i].repeat(k_test,1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        topk_idx = topk_idx.to(device)
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim
    return score_matrix_t2i

@torch.no_grad()
def get_eval_results(model, image_embeds, image_feats, k_test, device='cuda', caption=""):
    tokenizer = model.tokenizer
    caption = caption[:1000]
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    sims_matrix = image_embeds @ text_embed.t()
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = get_t2i_score_matrix(sims_matrix, image_feats, text_ids, text_atts, model, k_test)
    scores_t2i = score_matrix_t2i.cpu().numpy()
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]
    results = get_metric_results(ranks)
    return results

def get_scores(model, image_embeds, image_feats, k_test, device='cuda', caption="", aggregate_captions=[], update_sims=None):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    sims_matrix = image_embeds @ text_embed.t()
    sims_matrix = sims_matrix.t()
    if update_sims is not None:
        sims_matrix = sims_matrix * update_sims
        # sims_matrix = torch.clamp(sims_matrix, max=0.999)
    sims_matrix_list = [sims_matrix]

    for aggregate_caption in aggregate_captions:
        text_embed_aggregate = get_single_text_embed(model, aggregate_caption, device='cuda')
        sims_matrix2 = image_embeds @ text_embed_aggregate.t()
        sims_matrix2 = sims_matrix2.t()
        sims_matrix_list.append(sims_matrix2)

    sims_matrix_final = torch.vstack(sims_matrix_list)
    sims_matrix_final = torch.mean(sims_matrix_final, dim=0, keepdim=True)

    score_matrix_t2i = get_t2i_score_matrix(sims_matrix_final, image_feats, text_ids, text_atts, model, k_test)
    scores_t2i = score_matrix_t2i
    score = scores_t2i[0]

    return score

def get_inds_sa(model, image_embeds, image_feats, k_test, device='cuda', caption="", aggregate_captions=[], update_sims=None):
    score = get_scores(model, image_embeds, image_feats, k_test, device=device,
                       caption=caption, aggregate_captions=aggregate_captions, update_sims=update_sims)
    # inds = np.argsort(score)[::-1]
    return score

def get_inds_i2t(model, image_embeds, image_feats, k_test, device='cuda', caption=""):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    image_embeds = image_embeds.unsqueeze(dim=0)
    image_feats = image_feats.unsqueeze(dim=0)
    sims_matrix = image_embeds @ text_embed.t()
    score_matrix_i2t = get_i2t_score_matrix(sims_matrix, image_feats, text_ids, text_atts, model, sims_matrix.shape[1])
    scores_i2t = score_matrix_i2t.cpu().numpy()
    score = scores_i2t[0]
    inds = np.argsort(score)[::-1]
    return inds


def get_metric_results(ranks):
    mdR = np.median(ranks + 1)
    # Compute metrics
    vr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    vr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    vr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    vr_mean = (vr1 + vr5 + vr10) / 3
    return vr1, vr5, vr10, vr_mean, mdR

def get_map(similarity, g_pids, q_pids):
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

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
    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP,
                   }
    return eval_result