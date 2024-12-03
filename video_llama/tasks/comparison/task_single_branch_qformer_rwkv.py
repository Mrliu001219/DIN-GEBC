"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import json
import os

from video_llama.common.dist_utils import main_process
from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask
from video_llama.evaluation.gebc_evaluation.evaluation_utils import gebc_captioning_eval
from video_llama.common.logger import MetricLogger, SmoothedValue
from video_llama.datasets.data_utils import prepare_sample
import numpy as np
import torch


@registry.register_task("task_single_branch_qformer_rwkv")
class Task_Single_Branch_Qformer_RWKV(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, cfg, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.cfg = cfg

        self.report_metric = report_metric
        self.loss = 0
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        # 返回一个类
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
#############################################################################################################
        samples_all = {}
        samples_all['image_query_tokens'] = samples['image_query_tokens']
        samples_all['reference_points'] = samples['reference_points']
        samples_all['before_tokens'] = samples['before_tokens']
        samples_all['after_tokens'] = samples['after_tokens']
        samples_all['prompt_subject'] = samples['subject_prompt']
        samples_all['prompt_b_and_a'] = samples['status_before_prompt'] + samples['status_after_prompt']
        samples_all['boundary_id'] = samples['boundary_id'] + samples['boundary_id'] + samples['boundary_id']

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples_all,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        boundary_ids = samples_all["boundary_id"]
        length = len(boundary_ids)

        for caption, boundary_id in zip(captions[0:(length // 3)], boundary_ids[0:(length // 3)]):
            results.append({"caption": caption, "boundary_id": boundary_id, "type": "subject"})

        for caption, boundary_id in zip(captions[(length // 3):(2 * length // 3)],
                                        boundary_ids[(length // 3):(2 * length // 3)]):
            results.append({"caption": caption, "boundary_id": boundary_id, "type": "status_before"})

        for caption, boundary_id in zip(captions[(2 * length // 3):length], boundary_ids[(2 * length // 3):length]):
            results.append({"caption": caption, "boundary_id": boundary_id, "type": "status_after"})
#######################################################################################################
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=False,
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )
        # metrics = self._report_metrics(
        #     eval_result_file=eval_result_file, split_name=split_name
        # )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        # if 'test' in split_name:
        #     metrics = {"agg_metrics": 0.0}
        #     return metrics

        gt_file = self.cfg.datasets_cfg.gebc_builder_Qformer_RWKV_2optimizer_3feature.build_info.annotations.get(split_name).annotation_path
        #gt_file = self.cfg.datasets_cfg.gebc.build_info.annotations.get(split_name).annotation_path
        #gt_file = self.cfg.datasets_cfg.gebc_builder_Qformer_lead_RWKV.build_info.annotations.get(split_name).annotation_path

        scores = gebc_captioning_eval(eval_result_file, gt_file)
        #scores['agg_metrics'] = scores['overall_score']

        log_stats = {split_name: {k: v for k, v in scores.items()}}

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        return scores

    def train_step(self, model, samples):
        loss = model(samples)
        return loss
        #return loss_subject

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):

        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        #use_amp = scaler is not None
        use_amp = False

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            '''
            iter(data_loader)将数据加载器对象转换为一个可迭代对象，
            你可以使用next(data_loader)从中获取下一个批次的数据
            '''
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
                #loss_qformer= self.train_step(model=model, samples=samples)
            optimizer_single = optimizer

            optimizer_single.zero_grad()
            loss.backward()
            optimizer_single.step()


            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer_single.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        self.loss = metric_logger.meters['loss'].global_avg
        print(self.loss)

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }