import logging
import random
import json
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
# from video_llama.models.modeling_opt import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import AutoTokenizer, BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
import math
import numpy as np
from collections import OrderedDict

# rwkv
from RWKV_5.my_rwkv_init import rwkv_init

facebook_opt_2_7b_path = "/home/newdisk/yutao/model/facebook-opt-2.7b"
bert_base_uncased_path = "/home/newdisk/yutao/model/bert-base-uncased"


# from flamingo_pytorch import PerceiverResampler
@registry.register_model("Qformer_RWKV_2optimizer_3feature")
class VideoBLIP2OPT_Qformer_RWKV_2optimizer_3feature(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained(bert_base_uncased_path)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
            self,
            opt_model="",
            max_txt_len=30,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

            frozen_opt_proj=False,
            opt_proj_model='',
            max_frame_pos=32,
            # max_other_features_pos = 100,
            num_video_query_token=32,
            # num_other_feat_query_token = 6,
            q_former_hidden_size=4096,  # 768
            subject_q_former_hidden_size=768,
            # other_feat_total_size=768,
            rwkv_out_dim=4096,
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.q_former_hidden_size = q_former_hidden_size
        self.subject_q_former_hidden_size = subject_q_former_hidden_size
        logging.info('Loading OPT Tokenizer')
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        print(self.opt_tokenizer.padding_side)
        if self.opt_tokenizer.pad_token is None:
            self.opt_tokenizer.pad_token = self.opt_tokenizer.eos_token

        logging.info('Loading OPT Model')
        if self.low_resource:
            self.opt_model = OPTForCausalLM.from_pretrained(
                opt_model, torch_dtype=torch.float16, load_in_8bit=True, device_map={'': device_8bit}
            )
        else:
            self.opt_model = OPTForCausalLM.from_pretrained(
                opt_model, torch_dtype=torch.float16
            )

        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading OPT Done')

        print(self.opt_model.config.hidden_size)
        logging.info('Loading opt proj')
        self.opt_proj_Qformer = nn.Linear(
            self.q_former_hidden_size, self.opt_model.config.hidden_size
        )

        self.opt_proj_RWKV = nn.Linear(
            self.q_former_hidden_size, self.opt_model.config.hidden_size
        )

        # if opt_proj_model:
        #     print("load opt proj weight: {}".format(opt_proj_model))
        #     opt_proj_weight = torch.load(opt_proj_model, map_location="cpu")
        #     msg = self.opt_proj.load_state_dict(opt_proj_weight['model'], strict=False)

        if frozen_opt_proj:
            #  todo frozen  opt_proj
            for name, param in self.opt_proj_Qformer.named_parameters():
                param.requires_grad = False
            logging.info('OPT_proj_qformer is frozen')
            for name, param in self.opt_proj_RWKV.named_parameters():
                param.requires_grad = False
            logging.info('OPT_proj_rwkv is  frozen')
        else:
            for name, param in self.opt_proj_Qformer.named_parameters():
                param.requires_grad = True
            logging.info('OPT_proj_qformer is not frozen')
            for name, param in self.opt_proj_RWKV.named_parameters():
                param.requires_grad = True
            logging.info('OPT_proj_rwkv is not frozen')

        logging.info('Loading opt_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = self.opt_tokenizer.eos_token

        self.num_video_query_token = num_video_query_token

        # Init video Qformer
        #self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.q_former_hidden_size)
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token, \
                                                                              vision_width=self.subject_q_former_hidden_size,
                                                                              num_hidden_layers=2)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.temporal_pos_trans = nn.Linear(self.subject_q_former_hidden_size, self.subject_q_former_hidden_size)
        self.temporal_pos_trans_norm = nn.LayerNorm(self.subject_q_former_hidden_size)

        # video_qformer
        self.video_qformer_line = nn.Linear(self.subject_q_former_hidden_size, self.q_former_hidden_size)
        self.Qformer_block = nn.Sequential(OrderedDict([
            ('video_Qformer',self.video_Qformer),
            ('temporal_pos_trans',self.temporal_pos_trans),
            ('temporal_pos_trans_norm',self.temporal_pos_trans_norm),
            ('video_qformer_line',self.video_qformer_line),
            ('opt_proj_Qformer',self.opt_proj_Qformer)
        ]))


        # rwkv
        logging.info('Loading rwkv')
        self.rwkv_out_dim = rwkv_out_dim
        self.rwkv = rwkv_init()

        self.RWKV_block = nn.Sequential(OrderedDict([
            ('rwkv', self.rwkv),
            ('opt_proj_RWKV', self.opt_proj_RWKV)
        ]))


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.subject_q_former_hidden_size / 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats)
        # batch size, 2
        proposals = proposals.sigmoid() * scale
        # batch size, 2, 128
        pos = proposals[:, :, None] / dim_t
        # batch size, 2, 256
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos.view(pos.shape[0], 1, -1).float()
        return pos

    def encode_video_qformer(self, q_hidden_state, reference_points):
        with self.maybe_autocast():
            # add frame_pos embedding
            batch_size, time_length, _, _ = q_hidden_state.size()
            # 无position
            frame_hidden_state = q_hidden_state  # [16,12,32,768]
            #####################################################
            frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size,
                                                  t=time_length)
            # [16,384,768]
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(q_hidden_state.device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            # [16,32,768]
            # Embed boundary information,  batch size, 1, hidden_size
            reference_point_embed = self.Qformer_block.temporal_pos_trans_norm(
                self.Qformer_block.temporal_pos_trans(self.get_proposal_pos_embed(reference_points)))
            video_query_tokens = video_query_tokens + reference_point_embed
            # print(f"video_query_tokens:{video_query_tokens.shape}################################")
            video_query_output = self.Qformer_block.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state
            #########################################################################################
            video_hidden = self.Qformer_block.video_qformer_line(video_hidden)
            # print('qformer',video_hidden.shape)
            video_tokens = self.Qformer_block.opt_proj_Qformer(video_hidden)
            video_att_mask = torch.ones(video_tokens.size()[:-1], dtype=torch.long).to(video_tokens.device)
        return video_tokens, video_att_mask

    def encode_video_rwkv(self, q_hidden_state):
        with self.maybe_autocast():
            # add frame_pos embedding
            batch_size, time_length, _, _ = q_hidden_state.size()

            #frame_hidden_state = q_hidden_state  # [16,12,32,768]
            frame_hidden_state_1 = q_hidden_state
            frame_hidden_state_2 = torch.zeros((q_hidden_state.shape[0], q_hidden_state.shape[1], 32, 4096)).to(
                "cuda")

            # rwkv
            for i in range(32):
                frame_hidden_state_2[:, :, i, :] = self.RWKV_block.rwkv(frame_hidden_state_1[:, :, i, :])

            video_hidden = torch.mean(frame_hidden_state_2, dim=1)

            video_tokens = self.RWKV_block.opt_proj_RWKV(video_hidden)
            video_att_mask = torch.ones(video_tokens.size()[:-1], dtype=torch.long).to(video_tokens.device)
        return video_tokens, video_att_mask

    def prompt_wrap(self, video_embeds, atts_video, prompt):
        if prompt:
            batch_size = video_embeds.shape[0]
            # print(prompt)
            p_before_tokens = self.opt_tokenizer(
                'Video:', return_tensors="pt", add_special_tokens=False).to(video_embeds.device)
            p_after_tokens = self.opt_tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, padding='longest').to(video_embeds.device)
            p_before_embeds = self.opt_model.model.decoder.embed_tokens(p_before_tokens.input_ids).expand(batch_size,
                                                                                                          -1, -1)
            p_after_embeds = self.opt_model.model.decoder.embed_tokens(p_after_tokens.input_ids)
            p_after_attention_mask = p_after_tokens.attention_mask
            wrapped_video_embeds = torch.cat([p_before_embeds, video_embeds], dim=1)
            wrapped_atts_video = atts_video[:, :1].expand(-1, wrapped_video_embeds.shape[1])
            wrapped_video_embeds = torch.cat([wrapped_video_embeds, p_after_embeds], dim=1)
            wrapped_atts_video = torch.cat([wrapped_atts_video, p_after_attention_mask], dim=1)
            return wrapped_video_embeds, wrapped_atts_video
        else:
            return video_embeds, atts_video

    def get_loss_subject(self,video_embeds_subject, atts_video_subject, caption_subject):
        text_subject = [t + self.end_sym for t in caption_subject]

        to_regress_tokens_subject = self.opt_tokenizer(
            text_subject,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(video_embeds_subject.device)

        targets_subject = to_regress_tokens_subject.input_ids.masked_fill(
            to_regress_tokens_subject.input_ids == self.opt_tokenizer.pad_token_id, -100
        )

        empty_targets_subject = (
            torch.ones([video_embeds_subject.shape[0], atts_video_subject.shape[1] + 1],
                       dtype=torch.long).to(video_embeds_subject.device).fill_(-100)  # plus one for bos
        )
        targets_subject = torch.cat([empty_targets_subject, targets_subject], dim=1)

        batch_size = video_embeds_subject.shape[0]
        bos_subject = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens_subject.input_ids.dtype,
                                 device=to_regress_tokens_subject.input_ids.device) * self.opt_tokenizer.bos_token_id
        bos_embeds_subject = self.opt_model.model.decoder.embed_tokens(bos_subject)
        atts_bos_subject = atts_video_subject[:, :1]
        to_regress_embeds_subject = self.opt_model.model.decoder.embed_tokens(to_regress_tokens_subject.input_ids)

        inputs_embeds_subject = torch.cat([bos_embeds_subject, video_embeds_subject, to_regress_embeds_subject], dim=1)
        attention_mask_subject = torch.cat(
            [atts_bos_subject, atts_video_subject, to_regress_tokens_subject.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds_subject,
                attention_mask=attention_mask_subject,
                return_dict=True,
                labels=targets_subject,
            )
        loss_subject = outputs.loss
        return loss_subject

    def get_loss_before_and_after(self, video_embeds_b_and_a, atts_video_b_and_a, caption_b_and_a):
        text_b_and_a = [t + self.end_sym for t in caption_b_and_a]

        to_regress_tokens_b_and_a = self.opt_tokenizer(
            text_b_and_a,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(video_embeds_b_and_a.device)

        targets_b_and_a = to_regress_tokens_b_and_a.input_ids.masked_fill(
            to_regress_tokens_b_and_a.input_ids == self.opt_tokenizer.pad_token_id, -100
        )

        empty_targets_b_and_a = (
            torch.ones([video_embeds_b_and_a.shape[0], atts_video_b_and_a.shape[1] + 1],
                       dtype=torch.long).to(video_embeds_b_and_a.device).fill_(-100)  # plus one for bos
        )
        targets_b_and_a = torch.cat([empty_targets_b_and_a, targets_b_and_a], dim=1)

        batch_size = video_embeds_b_and_a.shape[0]
        bos_b_and_a = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens_b_and_a.input_ids.dtype,
                                 device=to_regress_tokens_b_and_a.input_ids.device) * self.opt_tokenizer.bos_token_id
        bos_embeds_b_and_a = self.opt_model.model.decoder.embed_tokens(bos_b_and_a)
        atts_bos_b_and_a = atts_video_b_and_a[:, :1]
        to_regress_embeds_b_and_a = self.opt_model.model.decoder.embed_tokens(to_regress_tokens_b_and_a.input_ids)

        inputs_embeds_b_and_a = torch.cat([bos_embeds_b_and_a, video_embeds_b_and_a, to_regress_embeds_b_and_a], dim=1)
        attention_mask_b_and_a = torch.cat(
            [atts_bos_b_and_a, atts_video_b_and_a, to_regress_tokens_b_and_a.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds_b_and_a,
                attention_mask=attention_mask_b_and_a,
                return_dict=True,
                labels=targets_b_and_a,
            )
        loss_before_and_after = outputs.loss
        return loss_before_and_after


    def forward(self, samples):
        self.opt_tokenizer.padding_side = "right"
        image_query_tokens = samples['image_query_tokens']
        reference_points = samples['reference_points']

        before_tokens = samples['before_tokens']
        after_tokens = samples['after_tokens']

        #subject
        video_embeds_subject, atts_video_subject = self.encode_video_qformer(image_query_tokens, reference_points)

        b_and_a_tokens = torch.cat([before_tokens, after_tokens], dim=0)
        #before_and_after
        video_embeds_b_and_a, atts_video_b_and_a= self.encode_video_rwkv(b_and_a_tokens)

        # loss_subject
        ########################################################################
        video_embeds_subject, atts_video_subject = self.prompt_wrap(video_embeds_subject, atts_video_subject,
                                                                    samples['subject_prompt'])
        caption_subject = samples['subject_caption']
        loss_subject = self.get_loss_subject(video_embeds_subject, atts_video_subject, caption_subject)
        ############################################################################################
        #loss_before_and_after
        ############################################################################################
        prompt_b_and_a = samples['status_before_prompt'] + samples['status_after_prompt']
        video_embeds_b_and_a, atts_video_b_and_a = self.prompt_wrap(video_embeds_b_and_a, atts_video_b_and_a,
                                                                    prompt_b_and_a)
        caption_b_and_a = samples['status_before_caption'] + samples['status_after_caption']
        loss_before_and_after = self.get_loss_before_and_after(video_embeds_b_and_a, atts_video_b_and_a, caption_b_and_a)
        ############################################################################################

        return {"loss_subject": loss_subject,
                "loss_before_and_after": loss_before_and_after}
        #return {"loss_subject": loss_subject}

    @torch.no_grad()
    def generate(
            self,
            samples_all,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        with self.maybe_autocast():
            image_query_tokens = samples_all['image_query_tokens']
            reference_points = samples_all['reference_points']
            before_tokens = samples_all['before_tokens']
            after_tokens = samples_all['after_tokens']

            #subject
            video_embeds_subject, atts_video_subject = self.encode_video_qformer(image_query_tokens, reference_points)

            b_and_a_tokens = torch.cat([before_tokens, after_tokens], dim=0)
            # before_and_after
            video_embeds_b_and_a, atts_video_b_and_a = self.encode_video_rwkv(b_and_a_tokens)
            # subject
            #################################################################################################################################
            prompt_subject = samples_all['prompt_subject']
            video_embeds_subject, atts_video_subject = self.prompt_wrap(video_embeds_subject, atts_video_subject,
                                                                        prompt_subject)

            batch_size_subject = video_embeds_subject.shape[0]
            bos_subject = torch.ones([batch_size_subject, 1],
                                     device=video_embeds_subject.device).long() * self.opt_tokenizer.bos_token_id
            bos_embeds_subject = self.opt_model.model.decoder.embed_tokens(bos_subject)
            atts_bos_subject = atts_video_subject[:, :1]

            inputs_embeds_subject = torch.cat([bos_embeds_subject, video_embeds_subject], dim=1)
            attention_mask_subject = torch.cat([atts_bos_subject, atts_video_subject], dim=1)

            outputs_subject = self.opt_model.generate(
                inputs_embeds=inputs_embeds_subject,
                attention_mask=attention_mask_subject,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.opt_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text_subject = self.opt_tokenizer.batch_decode(
                outputs_subject, skip_special_tokens=True
            )
            #####################################################################################################################################

            #before_and_after
            #####################################################################################################################################
            prompt_b_and_a = samples_all['prompt_b_and_a']
            video_embeds_b_and_a, atts_video_b_and_a = self.prompt_wrap(video_embeds_b_and_a, atts_video_b_and_a,
                                                                        prompt_b_and_a)

            batch_size_b_and_a = video_embeds_b_and_a.shape[0]
            bos_b_and_a = torch.ones([batch_size_b_and_a, 1],
                                     device=video_embeds_b_and_a.device).long() * self.opt_tokenizer.bos_token_id
            bos_embeds_b_and_a = self.opt_model.model.decoder.embed_tokens(bos_b_and_a)
            atts_bos_b_and_a = atts_video_b_and_a[:, :1]

            inputs_embeds_b_and_a = torch.cat([bos_embeds_b_and_a, video_embeds_b_and_a], dim=1)
            attention_mask_b_and_a = torch.cat([atts_bos_b_and_a, atts_video_b_and_a], dim=1)

            outputs_b_and_a = self.opt_model.generate(
                inputs_embeds=inputs_embeds_b_and_a,
                attention_mask=attention_mask_b_and_a,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.opt_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text_b_and_a = self.opt_tokenizer.batch_decode(
                outputs_b_and_a, skip_special_tokens=True
            )
            #####################################################################################################################################


            output_text_subject = [text for text in output_text_subject]
            output_text_b_and_a = [text for text in output_text_b_and_a]
            output_text = output_text_subject + output_text_b_and_a
        return output_text
        #return output_text_subject

    @classmethod
    def from_config(cls, cfg):
        q_former_hidden_size = cfg.get('q_former_hidden_size', 4096)  # 768
        opt_model = cfg.get("opt_model")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 30)
        end_sym = cfg.get("end_sym", '\n')
        frozen_opt_proj = cfg.get("frozen_opt_proj", False)
        opt_proj_model = cfg.get("opt_proj_model", '')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        num_video_query_token = cfg.get("num_video_query_token", 32)
        # max_other_features_pos = cfg.get("max_other_features_pos", 100)
        # other_feat_total_size = cfg.get("other_feat_total_size")
        # num_other_feat_query_token = cfg.get("num_other_feat_query_token", 6)

        model = cls(
            opt_model=opt_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            device_8bit=device_8bit,
            # the device of 8bit model should be set when loading and cannot be changed anymore.
            frozen_opt_proj=frozen_opt_proj,
            opt_proj_model=opt_proj_model,
            max_frame_pos=max_frame_pos,
            # num_other_feat_query_token = num_other_feat_query_token,
            num_video_query_token=num_video_query_token,
            # max_other_features_pos = max_other_features_pos,
            q_former_hidden_size=q_former_hidden_size,
            subject_q_former_hidden_size=768,
            # other_feat_total_size=other_feat_total_size,
            rwkv_out_dim=4096,
        )

        return model
