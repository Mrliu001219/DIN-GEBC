import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.my_datasets.gebc_dataset_Qformer_RWKV_2optimizer import GEBCDataset_Qformer_RWKV_2optimizer, EvalGEBCDataset_Qformer_RWKV_2optimizer


@registry.register_builder("gebc_builder_Qformer_RWKV_2optimizer")
class GEBCBuilder_Qformer_RWKV_2optimizer(BaseDatasetBuilder):
    train_dataset_cls = GEBCDataset_Qformer_RWKV_2optimizer
    eval_dataset_cls = EvalGEBCDataset_Qformer_RWKV_2optimizer

    DATASET_CONFIG_DICT = {"default": "configs/datasets/gebc/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        # self.build_processors()
        datasets = dict()
        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        split = "train"
        annotations = build_info.annotations
        video_info_path = build_info.video_info_path
        q_former_feature_folder = build_info.q_former_feature_folder
        # other_feature_names = build_info.other_feature_names
        # other_feature_folders = build_info.other_feature_folders
        max_seq_len = build_info.max_seq_len
        for split in ['train', 'val', 'test']:
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            annotation_path = annotations.get(split).annotation_path
            datasets[split] = dataset_cls(
                annotation_path=annotation_path,
                video_info_path=video_info_path,
                q_former_feature_folder=q_former_feature_folder,
                # other_feature_names=other_feature_names,
                # other_feature_folders=other_feature_folders,
                max_seq_len=max_seq_len,
            )
        return datasets