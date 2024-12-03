# import numpy as np
#
# # 指定.npy文件的路径
# file_path = 'data/features/eva_vit_g_q_former_tokens_16/9ZDhWd7mUHM.npy'
#
# # 使用np.load()读取.npy文件
# data = np.load(file_path)
#
# # 现在，'data'变量包含了.npy文件中的数据
# print(data.shape)

import argparse
import os
os.environ['TORCH_HOME'] = '.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '.cache'
os.environ['TRANSFORMERS_CACHE'] = '.cache'
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank, init_distributed_mode
from video_llama.common.logger import setup_logger
from video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from video_llama.common.registry import registry
from video_llama.common.utils import now

# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *


# def parse_args():
#     parser = argparse.ArgumentParser(description="Training")
#     #配置文件夹路径
#     parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#
#     args = parser.parse_args()#解析参数
#     # if 'LOCAL_RANK' not in os.environ:
#     #     os.environ['LOCAL_RANK'] = str(args.local_rank)
#
#     return args
#
#
# def setup_seeds(config):
#     seed = config.run_cfg.seed + get_rank()
#
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     cudnn.benchmark = False
#     cudnn.deterministic = True
#
#
# def get_runner_class(cfg):
#     """
#     Get runner class from config. Default to epoch-based runner.
#     """
#     runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
#
#     return runner_cls
#
# job_id = now()#时间
#
#
# cfg = Config(parse_args())
#
# init_distributed_mode(cfg.run_cfg)
#
# setup_seeds(cfg)
#
#     # set after init_distributed_mode() to only log on master.
# setup_logger()#配置一下输出函数，相当于print函数，但是比print安全
#
# #cfg.pretty_print()#输出Dataset相关信息
#
#     #  video_llama/tasks/boundary_captioning.py
# task = tasks.setup_task(cfg)#获得task的类
#     #  video_llama/tasks/base_task.py
# datasets = task.build_datasets(cfg)
# # print(datasets['gebc_builder_Qformer_lead_RWKV']['test'][0])
# # print("66666666666666666666666666666")
#
# #evaluate
# model = task.build_model(cfg)
# #model = nn.DataParallel(model).module
# runner = get_runner_class(cfg)(
#     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
# )
# runner.evaluate()

# length = 24
# list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# print(list[0:(length//3)])
# print(list[(length//3):(2*length//3)])
# print(list[(2*length//3):length])
import json


import numpy as np
from builtins import dict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
def split_pred(pred):
    pred_dict = {
        'subject': {},
        'status_before': {},
        'status_after': {}
    }
    # pred_dict = {
    #     'subject': {}
    # }
    for pred_item in pred:
        caption_type = pred_item['type']
        boundary_id = pred_item['boundary_id']
        caption = pred_item['caption']
        pred_dict[caption_type][boundary_id] = [caption]
    return pred_dict

def split_gt(gt):
    gt_dict = {
        'subject': {},
        'status_before': {},
        'status_after': {}
    }
    # gt_dict = {
    #     'subject': {},
    # }
    for _, video_anno in gt.items():
        for boundary in video_anno:
            boundary_id = boundary['boundary_id']
            subject = boundary['subject']
            status_before = boundary['status_before']
            status_after = boundary['status_after']
            gt_dict['subject'][boundary_id] = [subject]
            gt_dict['status_before'][boundary_id] = [status_before]
            gt_dict['status_after'][boundary_id] = [status_after]
    return gt_dict

class EvalCap:
    def __init__(self, pred_dict, gt_dict, df):
        self.evalBoundaries = []
        self.eval = dict()
        self.BoundariesToEval = dict()

        self.gts = gt_dict
        self.res = pred_dict
        self.df = df

    def tokenize(self):

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        self.gts = tokenizer.tokenize(self.gts)
        self.res = tokenizer.tokenize(self.res)

    def evaluate(self):
        self.tokenize()

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(self.df), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            # print(self.gts['8DoMr9pn4GQ_0'])
            # print(self.res['8DoMr9pn4GQ_0'])
            self.my_gts = dict()
            self.my_res = dict()
            self.my_gts['8DoMr9pn4GQ_0'] = ['human hands','human hands','human hands']
            self.my_gts['zSDvE4EJj-Q_1'] = ['the man in red t-shirt and blue shorts','the man in red t-shirt and blue shorts']
            self.my_res['8DoMr9pn4GQ_0'] = self.res['8DoMr9pn4GQ_0']
            self.my_res['zSDvE4EJj-Q_1'] = self.res['zSDvE4EJj-Q_1']

            # st
            score, scores = scorer.compute_score(self.my_gts, self.my_res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setBoundaryToEvalBoundaries(scs, self.gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setBoundaryToEvalBoundaries(scores, self.gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalBoundaries()

    def setEval(self, score, method):
        self.eval[method] = score

    def setBoundaryToEvalBoundaries(self, scores, b_ids, method):
        for b_id, score in zip(b_ids, scores):
            if not b_id in self.BoundariesToEval:
                self.BoundariesToEval[b_id] = dict()
                self.BoundariesToEval[b_id]["boundary_id"] = b_id
            self.BoundariesToEval[b_id][method] = score

    def setEvalBoundaries(self):
        self.evalBoundaries = [eval for imgId, eval in self.BoundariesToEval.items()]

def evaluate_on_caption(pred_dict, gt_dict, outfile=None):
    def _convert(d):
        for key, captions in d.items():
            temp = []
            for caption in captions:
                temp.append({'caption':caption})
            d[key] = temp
        return d
    pred_dict = _convert(pred_dict)
    gt_dict = _convert(gt_dict)

    Eval = EvalCap(pred_dict, gt_dict, 'corpus')

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    Eval.evaluate()
    result = Eval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result

def gebc_captioning_eval(pred_file_path, gt_file_path):
    with open(pred_file_path, 'r') as f:
        predictions = json.load(f)
    with open(gt_file_path, 'r') as f:
        groundtruths = json.load(f)
    pred_dict = split_pred(predictions)
    gt_dict = split_gt(groundtruths)
    res_pred_sub = evaluate_on_caption(pred_dict['subject'], gt_dict['subject'])
    print(pred_dict['subject'].keys())
    print(res_pred_sub)

pred_file_path = "video_llama/output/video_blip2_opt_highest_f1_12frame/puar_common_lead_20epoch/result/val_epoch9.json"
gt_file_path = "data/annotations/valset_highest_f1.json"
gebc_captioning_eval(pred_file_path,gt_file_path)