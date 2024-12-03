import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.my_datasets.msvd_msrvtt_dataset import MSVD_MSRVTT_Dataset, Eval_MSVD_MSRVTT_Dataset


@registry.register_builder("msvd_msrvtt_builder")
class MSVD_MSRVTT_Builder(BaseDatasetBuilder):
    train_dataset_cls = MSVD_MSRVTT_Dataset
    eval_dataset_cls = Eval_MSVD_MSRVTT_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/gebc/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):

        datasets = dict()
        build_info = self.config.build_info

        annotations = build_info.annotations
        q_former_feature_folder = build_info.q_former_feature_folder
        max_seq_len = build_info.max_seq_len
        for split in ['train', 'val', 'test']:
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            annotation_path = annotations.get(split).annotation_path
            datasets[split] = dataset_cls(
                annotation_path=annotation_path,
                q_former_feature_folder=q_former_feature_folder,
                max_seq_len=max_seq_len,
            )
        return datasets