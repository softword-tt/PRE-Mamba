import os
import numpy as np
import re

import random

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset
from copy import deepcopy

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or dirpath.split('/')[-1] in selected_file_list:
                yield os.path.abspath(os.path.join(dirpath, f))
                
                
@DATASETS.register_module()
class SPACMultiScansDataset(DefaultMultiScansDataset):
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

    def prepare_test_data(self, idx):
        # load data
        # data_dict = self.get_data(idx)
        data_dict = self.get_multi_data(idx)
        data_dict = self.transform(data_dict)
        data_path = self.data_list[idx % len(self.data_list)]
        [fold_name, file_name] = data_path.split("/")[-2:]
        file_name = file_name.split(".")[0]
        name = fold_name + "+" + file_name
        result_dict = dict(
            segment=data_dict.pop("segment"), name=name, tn=data_dict["tn"],
        )

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")
            result_dict["origin_tn"] = data_dict.pop("origin_tn")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.scale2kitti is not None:
                data = self.scale2kitti(data)
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
    

    def get_data_list(self):
        split2seq = dict(
            train = ["t1_Rain_01", "t1_Rain_02", "t1_Rain_03",       # demo0301_pro
                    #  "t2_Rain_01", "t2_Rain_02", "t2_Rain_03", 
                    "t3_Rain_01", "t3_Rain_02", "t3_Rain_03", 
                    #  "t4_Rain_01", "t4_Rain_02", "t4_Rain_03", 
                    "t5_Rain_01", "t5_Rain_02", "t5_Rain_03", 
                    "t6_Rain_01", "t6_Rain_02", "t6_Rain_03", 
                    #  "t7_Rain_01", "t7_Rain_02", "t7_Rain_03", "t7_Rain_04", 
                    "t8_Rain_01", "t8_Rain_02", "t8_Rain_03"],
            val = ['b1_Rain','b2_Rain','b3_Rain'],
            test = ['b2_Rain']
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
        data_list += absoluteFilePaths(self.data_root, seq_list)
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

            segment_path = scan_path.replace("SPAC-dataset-merge", "SPAC-dataset-event").replace("events3", "gt")
            segment_path = re.sub(r"([a-zA-Z]\d+)_Rain(_\d+)?", r"\1_GT", segment_path)
            
            with np.load(segment_path) as segment_data:
                segment_x = segment_data['x']
                segment_y = segment_data['y']
                segment_t = segment_data['t']
                segment_coord = np.stack((segment_x, segment_y, segment_t), axis=-1)
                if len(segment_coord) == 0:
                    print(f"Segment data is empty: {segment_path}")
                    continue

            labels = np.ones(len(coord), dtype=np.int32)

            segment_set = set(map(tuple, segment_coord)) 
            for idx, point in enumerate(coord):
                if tuple(point) in segment_set:
                    labels[idx] = 0

            t_normalized = (t - t.min()) / (t.max() - t.min())  

            coord = np.column_stack((x, y, t_normalized))  

            tn.append(np.ones(coord.shape[0]) * i)  

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


@DATASETS.register_module()
class SPACMultiScansDataset_V2(SPACMultiScansDataset):
    def __init__(self, split="train", data_root="data", gather_num=6, scan_modulation=False, transform=None, test_mode=False, test_cfg=None, windows_stride=None, loop=1, ignore_index=-1):
        super().__init__(split, data_root, gather_num, scan_modulation, transform, test_mode, test_cfg, windows_stride, loop, ignore_index)
    
    def get_data_list(self):
        data_list = []
        data_list += absoluteFilePaths(self.data_root)
        print(data_list)
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

            # Read `.npz` data
            with np.load(scan_path) as data:
                x = data['x'] 
                y = data['y'] 
                t = data['t']
                p = data['p']
                labels = data['labels']
                if len(x) == 0:
                    print(f"No points in file: {scan_path}")
                    continue
                coord = np.stack((x, y, t), axis=-1)
                strength = p.reshape(-1, 1)

            t_normalized = (t - t.min()) / (t.max() - t.min())  

            coord = np.column_stack((x, y, t_normalized))  

            tn.append(np.ones(coord.shape[0]) * i)  

            gather_coord.append(coord)
            gather_strength.append(strength)
            gather_segment.append(labels)
            multi_scan_path.append(scan_path)

        data_dict = dict(coord=np.concatenate(gather_coord), strength=np.concatenate(gather_strength),
                        segment=np.concatenate(gather_segment), tn=np.expand_dims(np.concatenate(tn), axis=1))
        return data_dict

@DATASETS.register_module()
class SPACMultiScansDataset_V3(SPACMultiScansDataset):
    def __init__(self, split="train", data_root="data", gather_num=6, scan_modulation=False, transform=None, test_mode=False, test_cfg=None, windows_stride=None, loop=1, ignore_index=-1):
        super().__init__(split, data_root, gather_num, scan_modulation, transform, test_mode, test_cfg, windows_stride, loop, ignore_index)
    
    def get_data_list(self):
        split2seq = dict(
            # test=['a1_GT', 'a2_GT', 'a3_GT', 'a4_GT', 
            #       'b1_GT', 'b2_GT', 'b3_GT', 'b4_GT', 
            #       't1_GT', 't2_GT', 't3_GT', 't4_GT', 't5_GT', 't6_GT', 't7_GT', 't8_GT'],
            # train = ['t1_Rain_01'],
            train = ['b2_Rain'],
            val = ['b2_Rain'],
            test = ['b2_Rain']
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
        data_list += absoluteFilePaths(self.data_root, seq_list)
        
        print(data_list)
        return data_list
    