"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from video_llama.common.dist_utils import main_process
from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask
from video_llama.evaluation.gebc_evaluation.evaluation_utils import gebc_captioning_eval
import numpy as np
import torch

@registry.register_task("boundary_captioning")
class BoundaryCaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, cfg, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.cfg = cfg

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        #返回一个类
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            cfg=cfg,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []
        # subject的dataset
        samples_1_indices = [i for i, value in enumerate(samples['caption_type']) if value == 'subject']
        samples_1 = {key: [value[i] for i in samples_1_indices] for key, value in samples.items()}
        numpy_query_1 = np.array([item.cpu().detach().numpy() for item in samples_1['image_query_tokens']])
        samples_1['image_query_tokens'] = torch.tensor(numpy_query_1).cuda()
        numpy_reference_1 = np.array([item.cpu().detach().numpy() for item in samples_1['reference_points']])
        samples_1['reference_points'] = torch.tensor(numpy_reference_1).cuda()

        # before和after的dataset
        samples_2_indices = [i for i, value in enumerate(samples['caption_type']) if value != 'subject']
        samples_2 = {key: [value[i] for i in samples_2_indices] for key, value in samples.items()}
        numpy_query_2 = np.array([item.cpu().detach().numpy() for item in samples_2['image_query_tokens']])
        samples_2['image_query_tokens'] = torch.tensor(numpy_query_2).cuda()
        numpy_reference_2 = np.array([item.cpu().detach().numpy() for item in samples_2['reference_points']])
        samples_2['reference_points'] = torch.tensor(numpy_reference_2).cuda()

        samples_all = {}
        samples_all['prompt'] = samples_1['prompt'] + samples_2['prompt']
        samples_all['boundary_id'] =  samples_1['boundary_id'] + samples_2['boundary_id']
        samples_all['caption_type'] = samples_1['caption_type'] + samples_2['caption_type']

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples_1,
            samples_2,
            samples_all,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        boundary_ids = samples_all["boundary_id"]
        types = samples_all["caption_type"]
        for caption, boundary_id, type in zip(captions, boundary_ids, types):
            results.append({"caption": caption, "boundary_id": boundary_id, "type": type})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=False,
        )

        if self.report_metric and split_name != 'test':
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        if 'test' in split_name:
            metrics = {"agg_metrics": 0.0}
            return metrics
        
        gt_file = self.cfg.datasets_cfg.gebc.build_info.annotations.get(split_name).annotation_path
        

        scores = gebc_captioning_eval(eval_result_file, gt_file)
        scores['agg_metrics'] = scores['overall_score']
        
        log_stats = {split_name: {k: v for k, v in scores.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        return scores
