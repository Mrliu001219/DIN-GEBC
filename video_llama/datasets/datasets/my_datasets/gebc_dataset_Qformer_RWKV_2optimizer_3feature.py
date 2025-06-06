import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
import json
import numpy as np
import pickle
import torch
from scipy.interpolate import interp1d


def read_file(path, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'pkl':
            with open(path, 'rb') as f:
                feats = pickle.load(f)
        else:
            raise NotImplementedError

        padding = False
    else:
        raise FileNotFoundError('{} not exists'.format(path))
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0
    if vf_type == 'q_former_tokens':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.npy')
    elif vf_type == 'intern_video':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'omni':
        feat_dim = 1536
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'clip':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    else:
        raise AssertionError('feature type error: {}'.format(vf_type))
    feats, padding = read_file(path, MEAN, VAR, data_norm)

    assert feats.shape[-1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats


def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new


def build_prompt(boundary_type, caption_type):
    prompt = 'This video describes the {}.'.format(boundary_type.lower())
    if caption_type == 'subject':
        prompt += 'The subject is'
    elif caption_type == 'status_before':
        prompt += 'Status before change is'
    else:
        prompt += 'Status after change is'
    return prompt


class GEBCDataset_Qformer_RWKV_2optimizer_3feature(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, max_seq_len):
        self.q_former_feature_folder = q_former_feature_folder
        # self.other_feature_names = other_feature_names
        # self.other_feature_folders = other_feature_folders
        self.max_seq_len = max_seq_len
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        with open(video_info_path, 'r') as f:
            self.video_info = json.load(f)
        self._load_annotations(annotation_path)

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_boundaries in data.items():
            if not key in self.video_info:
                print('missing key:', key)
                continue
            duration = self.video_info[key]
            for video_anno in video_boundaries:
                boundary_duration = float(video_anno['next_timestamp']) - float(video_anno['prev_timestamp'])

                GEBC_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'prev_timestamp': video_anno['prev_timestamp'],
                    'timestamp': video_anno['timestamp'],
                    'next_timestamp': video_anno['next_timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'subject_caption': video_anno['subject'],
                    'status_before_caption': video_anno['status_before'],
                    'status_after_caption': video_anno['status_after']
                }

                self.annotation.append(GEBC_data)

    def __getitem__(self, index):
        item_data = self.annotation[index]
        # Prepare boundary information
        boundary_duration, duration = item_data['boundary_duration'],item_data['duration']

        prev_timestamp, boundary_timestamp, next_timestamp= item_data['prev_timestamp'], item_data['timestamp'],item_data['next_timestamp']

        reference_point = np.array([boundary_timestamp / duration, boundary_duration / duration])
        reference_point = torch.from_numpy(reference_point)
        # Prepare prompt
        boundary_type = item_data['label']
        subject_prompt = build_prompt(boundary_type, "subject")
        status_before_prompt = build_prompt(boundary_type, "status_before")
        status_after_prompt = build_prompt(boundary_type, "status_after")
        # Load caption
        subject_caption = item_data['subject_caption']
        status_before_caption = item_data['status_before_caption']
        status_after_caption = item_data['status_after_caption']
        # Load feature
        source_tokens = get_feats(item_data['boundary_id'], 'q_former_tokens', self.q_former_feature_folder)
        #subject
        #q_former_tokens = source_tokens[::2]
        #q_former_tokens = np.zeros_like(source_tokens)
        #q_former_tokens = source_tokens.copy()
        q_former_tokens = torch.from_numpy(source_tokens)  # (t,q,h)
        if (boundary_timestamp > next_timestamp):
            boundary_timestamp = max(next_timestamp - 2, 0)
        if (prev_timestamp > boundary_timestamp):
            prev_timestamp = max(boundary_timestamp - 2, 0)
        #before
        #(12,32,768)
        before_tokens = np.zeros_like(source_tokens)
        #before_tokens = np.zeros((source_tokens.shape[0],source_tokens.shape[1],source_tokens.shape[2]))
        before_start = max(int(source_tokens.shape[0] * prev_timestamp/duration -0.5) - 1, 0)
        before_end = min(int(source_tokens.shape[0] * boundary_timestamp/duration + 0.5), source_tokens.shape[0])
        # if before_end - before_start > 12:
        #     before_end = before_end - (before_end - before_start - 12)
        before_tokens[:before_end - before_start] = source_tokens[before_start:before_end]
        before_tokens = torch.from_numpy(before_tokens)  # (t,q,h)

        #after

        after_tokens = np.zeros_like(source_tokens)
        #after_tokens = np.zeros((source_tokens.shape[0], source_tokens.shape[1], source_tokens.shape[2]))
        after_start = max(int(source_tokens.shape[0] * boundary_timestamp / duration -0.5) - 1, 0)
        after_end = min(int(source_tokens.shape[0] * next_timestamp / duration + 0.5), source_tokens.shape[0])
        # if after_end - after_start > 12:
        #     after_end = after_end - (after_end - after_start - 12)
        #print(after_start,after_end,after_end-after_start,boundary_timestamp,next_timestamp,duration,source_tokens.shape[0])
        after_tokens[:after_end - after_start] = source_tokens[after_start:after_end]
        after_tokens = torch.from_numpy(after_tokens)  # (t,q,h)

        # load  other feature
        # other_features_list = [] # (a,t,h), a is the number of other features
        # for i, folder in enumerate(self.other_feature_folders):
        #     other_feature = get_feats(item_data['boundary_id'], self.other_feature_names[i], folder) # (t,h)
        #     other_features_list.append(other_feature)

        return {
            'image_query_tokens': q_former_tokens,
            # 'other_features_list': other_features_list,
            'reference_points': reference_point,
            'before_tokens': before_tokens,
            'after_tokens': after_tokens,

            'subject_prompt': subject_prompt,
            'status_before_prompt': status_before_prompt,
            'status_after_prompt': status_after_prompt,

            'subject_caption': subject_caption,
            'status_before_caption': status_before_caption,
            'status_after_caption': status_after_caption,
            'boundary_id': item_data['boundary_id'],
        }



class EvalGEBCDataset_Qformer_RWKV_2optimizer_3feature(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, max_seq_len):
        self.q_former_feature_folder = q_former_feature_folder
        # self.other_feature_names = other_feature_names
        # self.other_feature_folders = other_feature_folders
        self.max_seq_len = max_seq_len
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        with open(video_info_path, 'r') as f:
            self.video_info = json.load(f)
        self._load_annotations(annotation_path)

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_boundaries in data.items():
            if not key in self.video_info:
                print('missing key:', key)
                continue
            duration = self.video_info[key]
            for video_anno in video_boundaries:
                boundary_duration = float(video_anno['next_timestamp']) - float(video_anno['prev_timestamp'])

                GEBC_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'prev_timestamp': video_anno['prev_timestamp'],
                    'timestamp': video_anno['timestamp'],
                    'next_timestamp': video_anno['next_timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                }

                self.annotation.append(GEBC_data)

    def __getitem__(self, index):
        item_data = self.annotation[index]

        # Prepare boundary information
        boundary_duration, duration = item_data['boundary_duration'], item_data['duration']
        prev_timestamp, boundary_timestamp, next_timestamp = item_data['prev_timestamp'], item_data['timestamp'], \
        item_data['next_timestamp']

        reference_point = np.array([boundary_timestamp /duration, boundary_duration /duration])
        reference_point = torch.from_numpy(reference_point)
        # Prepare prompt
        boundary_type = item_data['label']
        subject_prompt = build_prompt(boundary_type, "subject")
        status_before_prompt = build_prompt(boundary_type, "status_before")
        status_after_prompt = build_prompt(boundary_type, "status_after")

        # Load feature
        source_tokens = get_feats(item_data['boundary_id'], 'q_former_tokens', self.q_former_feature_folder)
        # subject
        #q_former_tokens = source_tokens[::2]
        #q_former_tokens = np.zeros_like(source_tokens)
        #q_former_tokens = source_tokens.copy()
        q_former_tokens = torch.from_numpy(source_tokens)  # (t,q,h)

        # before
        # (12,32,768)
        before_tokens = np.zeros_like(source_tokens)
        #before_tokens = np.zeros((source_tokens.shape[0], source_tokens.shape[1], source_tokens.shape[2]))
        before_start = max(int(source_tokens.shape[0] * prev_timestamp / duration -0.5) - 1, 0)
        before_end = min(int(source_tokens.shape[0] * boundary_timestamp / duration + 0.5), source_tokens.shape[0])
        # if before_end - before_start > 12:
        #     before_end = before_end - (before_end - before_start - 12)
        before_tokens[:before_end - before_start] = source_tokens[before_start:before_end]
        before_tokens = torch.from_numpy(before_tokens)  # (t,q,h)

        # after
        after_tokens = np.zeros_like(source_tokens)
        #after_tokens = np.zeros((source_tokens.shape[0], source_tokens.shape[1], source_tokens.shape[2]))
        after_start = max(int(source_tokens.shape[0] * boundary_timestamp / duration -0.5) - 1, 0)
        after_end = min(int(source_tokens.shape[0] * next_timestamp / duration + 0.5), source_tokens.shape[0])
        # if after_end - after_start > 12:
        #     after_end = after_end - (after_end - after_start - 12)
        after_tokens[:after_end - after_start] = source_tokens[after_start:after_end]
        after_tokens = torch.from_numpy(after_tokens)  # (t,q,h)

        # load  other feature
        # other_features_list = [] # (a,t,h), a is the number of other features
        # for i, folder in enumerate(self.other_feature_folders):
        #     other_feature = get_feats(item_data['boundary_id'], self.other_feature_names[i], folder) # (t,h)
        #     other_features_list.append(other_feature)
        return {
            'image_query_tokens': q_former_tokens,
            # 'other_features_list': other_features_list,
            'reference_points': reference_point,
            'before_tokens': before_tokens,
            'after_tokens': after_tokens,

            'subject_prompt': subject_prompt,
            'status_before_prompt': status_before_prompt,
            'status_after_prompt': status_after_prompt,

            'boundary_id': item_data['boundary_id'],
        }

