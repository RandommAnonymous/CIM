import torch
from models.CIM import CIM, load_pretrained, AllGather
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import viridis
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import norm
import random
class CIM_Retrieval(CIM):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config['mlm'])
        if not self.pa100k_only_img_classifier:
            self.mlm = config['mlm']
            self.pa100k = config['pa100k']
            self.total_epoch = config['schedular']['epochs']
            self.gpt = config['gpt']
            if not self.pa100k:
                self.eda = config['eda']
            if ('attr' in config.keys()) and config['attr']:
                self.attr = True
                self.t = config['t']
            else:
                self.attr = False

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("vision_encoder missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, attr_text_ids=None, attr_text_atts=None, attr_text_ids_masked=None,
                attr_masked_pos=None, attr_masked_ids=None, label=None, text_ids_eda=None, text_atts_eda=None, 
                cur_epoch=None, diffusion=None, schedule_sampler=None, gpt_input=None):
        
        add_loss = dict()
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                            text_embeds, text_atts, text_feat, idx=idx)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos,
                                        masked_ids)
        add_loss.update({'loss_mlm': loss_mlm})
        if self.attr:

            attr_text_embeds = self.get_text_embeds(attr_text_ids, attr_text_atts)
            attr_text_feat = self.get_features(text_embeds=attr_text_embeds)

            attr_loss_itc = self.get_contrastive_loss_attr(image_feat, attr_text_feat, label)
            attr_loss_itm = self.get_matching_loss_attr(image_embeds, image_atts, attr_text_embeds, attr_text_atts,
                                                        label)


            attr_loss_mlm = self.get_mlm_loss_attr(attr_text_ids_masked, attr_text_atts, image_embeds, image_atts,
                                                    attr_masked_pos, attr_masked_ids, label)
            loss_attr = self.t * (attr_loss_itc + attr_loss_itm + attr_loss_mlm) / 3
            add_loss.update({'loss_attr': loss_attr})

        
        # eda
        if self.eda:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_features(text_embeds=text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx=idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx=idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda
        add_loss.update({'loss_itc': loss_itc})
        add_loss.update({'loss_itm': loss_itm})
        
        if self.gpt:
            gpt_ids = gpt_input.input_ids
            gpt_atts = gpt_input.attention_mask
            gpt_feat = self.text_proj(self.get_text_embeds(gpt_ids, gpt_atts))[:, 0, :]

            co_gpt_feat1, co_text_feat = self.gpt_caption(gpt_feat, text_feat)
            loss_gtc = self.get_contrastive_loss(co_gpt_feat1, co_text_feat, idx=idx)

            co_gpt_feats2, co_image_feat = self.img_caption(gpt_feat, image_feat)
            loss_g2i = self.get_contrastive_loss(co_gpt_feats2, co_image_feat, idx=idx)
            add_loss.update({'loss_gpt_contrasive': loss_g2i + loss_gtc})


            
                
        return add_loss


