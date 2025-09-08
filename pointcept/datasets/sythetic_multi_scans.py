import os
import numpy as np
import re

import random

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or f.split('.')[0] in selected_file_list:
                yield os.path.abspath(os.path.join(dirpath, f))

@DATASETS.register_module()
class EventRainMultiScansDataset(DefaultMultiScansDataset):
    def __init__(
        self,
        split="train",
        data_root="data",
        gather_num=6,
        scan_modulation=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        windows_stride=None,
        loop=1,
        ignore_index=-1,
    ):
        self.gather_num = gather_num
        self.scan_modulation = scan_modulation
        self.ignore_index = ignore_index
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            gather_num=gather_num,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        self.windows_stride = windows_stride

    def get_data_list(self):
        split2seq = dict(
            train=[1, 10, 15, 17, 25, 30, 40, 60, 75, 100, 175, 200],
            val=[50],
            # test=[5, 20, 50, 80, 125, 150],
            test=[5],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        self.poses = {}
        for seq in seq_list:
            seq = str(seq) + "mm"
            seq_folder = os.path.join(self.data_root, "sythetic", "merge_data")
            if self.split == "train":
                data_list += absoluteFilePaths(os.path.join(seq_folder, seq))
                print(f"Processing all frames in Seq {seq} without filtering.")
            else:
                data_list += absoluteFilePaths(os.path.join(seq_folder, seq))
        return data_list

    def get_multi_data(self, idx):
        cur_data_path = self.data_list[idx % len(self.data_list)]

        multi_scan_path, gather_coord, gather_strength, gather_segment = [], [], [], []
        seq, _, file_name = cur_data_path.split('/')[-3:]

        cur_scan_index = int(file_name.split('.')[0])

        tn = []
        modulation = 1
        if self.scan_modulation:
            scan_modulation_prob = random.random()
            if 0.5 < scan_modulation_prob <= 0.75:
                modulation = 2
            elif scan_modulation_prob > 0.75:
                modulation = 3
            if self.split != "train":
                modulation = 3

        for i in range(self.gather_num):
            last_scan_index = cur_scan_index - modulation * i
            last_scan_index = max(0, last_scan_index)
            scan_path = cur_data_path.replace(cur_data_path.split("/")[-1], f"{str(last_scan_index).zfill(10)}.npz")
            if not os.path.exists(scan_path):
                print(f"File {scan_path} does not exist, skipping...")
                continue

            with np.load(scan_path) as data:
                x = data['x'] 
                y = data['y'] 
                t = data['t']
                p = data['p']
                if len(x) == 0:
                    print(f"No points in file: {scan_path}")
                    continue

                coord = np.stack((x, y, t), axis=-1)
                strength = p.reshape(-1, 1)

            segment_path = scan_path.replace("merge_data", "raw_data")
            segment_path = re.sub(r"/\d+mm/", "/", segment_path)

            with np.load(segment_path) as segment_data:
                segment_x = segment_data['x']
                segment_y = segment_data['y']
                segment_t = segment_data['t']
                segment_coord = np.stack((segment_x, segment_y, segment_t), axis=-1)
                if len(segment_coord) == 0:
                    print(f"Segment data is empty: {segment_path}")
                    continue

            labels = np.ones(len(coord), dtype=np.int32)
            # Assign label 0 to points in the segment
            segment_set = set(map(tuple, segment_coord))  # Use a set to accelerate search
            for idx, point in enumerate(coord):
                if tuple(point) in segment_set:
                    labels[idx] = 0

            t_normalized = (t - t.min()) / (t.max() - t.min())  # Normalize t to the range [0, 1]

            # Add the normalized time t to coord.
            coord = np.column_stack((x, y, t_normalized))  # 组合 (x, y, t_normalized)

            # Add time window index
            tn.append(np.ones(coord.shape[0]) * i)  # i 为窗口索引

            gather_coord.append(coord)
            gather_strength.append(strength)
            gather_segment.append(labels)
            multi_scan_path.append(scan_path)

        data_dict = dict(coord=np.concatenate(gather_coord), strength=np.concatenate(gather_strength),
                        segment=np.concatenate(gather_segment), tn=np.expand_dims(np.concatenate(tn), axis=1))
        return data_dict

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        '''     
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        '''

        segment_path = data_path.replace("velodyne", "segments").replace(".bin", ".segment")
        if os.path.exists(segment_path):
            with open(segment_path, "rb") as a:
                segment = np.fromfile(a, dtype=np.float32).reshape(-1, 4)
        else:
            segment = np.zeros(scan.shape).astype(np.float32)

        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"

        return data_name

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 0,
            1: 1,
            2: 250,
            3: 251,
        }

        return learning_map_inv
