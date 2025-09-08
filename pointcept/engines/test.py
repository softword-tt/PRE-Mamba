import math
import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from scipy.spatial import cKDTree
from torch.utils.tensorboard import SummaryWriter
from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False):
        super().__init__(cfg, model=model, test_loader=test_loader, verbose=verbose)
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.save_path, 'testlogs'))
        self.shape = cfg.size
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        moving_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                (self.cfg.data.test.type == "SPACMultiScansDataset" or
                 self.cfg.data.test.type == "SPACMultiScansDataset_V2" or
                 self.cfg.data.test.type == "EventRainMultiScansDataset" or 
                 self.cfg.data.test.type == "EVK4MultiScansDataset") and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))

        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            tn = data_dict.pop("tn")
            final_scan_mask = tn.squeeze(1) == 0

            pred_save_path = os.path.join(save_path, "{}.npy".format(data_name))

            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            coord = torch.zeros((segment.size, 3)).cuda()
            p = torch.zeros((segment.size, 1)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    coord_part = input_dict["coord"]
                    p_part = input_dict["feat"][:, 3].unsqueeze(1)
                    # print(f"Pred part shape: {pred_part.shape}")
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        coord[idx_part[bs:be], :] = coord_part[bs:be]
                        p[idx_part[bs:be], :] = p_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i+1,
                        batch_num=len(fragment_list),
                    )
                )
                if i == 0:  # Only visualize the first batch (can be adjusted)
                    coord_np = coord.cpu().numpy()
                    pred_np = pred.cpu().numpy() 
                    p_np = p.cpu().numpy()          # Positive and negative events of the original point
                    p_np = p.squeeze()


                    x, y = coord_np[:, 0], coord_np[:, 1]
                    # Define the render function
                    def render(x, y, p, mask, shape):
                        # print(f"x shape: {x.shape}, y shape: {y.shape}, p shape: {p.shape}", f"mask shape: {mask.shape}")
                        img = np.full(shape=tuple(shape) + (3,), fill_value=255, dtype="uint8")  
                        # valid = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1]) & mask  
                        valid = mask
                        x, y, p = x[valid], y[valid], p[valid]
                        x = x.astype(int)
                        y = y.astype(int)
                        img[y, x, :] = 0 
                        img[y, x, 0] = 255 * (p > 0)  # Red Channel: Positive Event
                        img[y, x, 1] = 255 * (p < 0)  # Green channel: adverse events
                        return img

                    # print(f"p_np type: {p_np.dtype}")
                    p_np = p_np if isinstance(p_np, np.ndarray) else p_np.cpu().numpy()
                    original_img = render(x, y, p_np, np.ones_like(p_np, dtype=bool), self.shape)

                    gt_mask = segment == 0
                    gt_img = render(x, y, p_np, gt_mask, self.shape)

                    gt_mask_rain = segment == 1
                    gt_img_rain = render(x, y, p_np, gt_mask_rain, self.shape)

                    max_indices = np.argmax(pred_np, axis=1)  
                    # print(f"max_indices: {max_indices}")

                    pred_mask = max_indices == 0
                    pred_img = render(x, y, p_np, pred_mask, self.shape)

                    # Generate prediction result images (only points where label == 0)
                    pred_mask_rain = max_indices == 1 # Add debugging information
                    pred_img_rain = render(x, y, p_np, pred_mask_rain, self.shape)

                    # Save to TensorBoard
                    original_tensor = torch.tensor(original_img).permute(2, 0, 1) / 255.0  # Convert to Tensor
                    gt_tensor = torch.tensor(gt_img).permute(2, 0, 1) / 255.0
                    pred_tensor = torch.tensor(pred_img).permute(2, 0, 1) / 255.0
                    pred_tensor_rain = torch.tensor(pred_img_rain).permute(2, 0, 1) / 255.0
                    gt_img_rain = torch.tensor(gt_img_rain).permute(2, 0, 1) / 255.0
                    if self.writer is not None:
                        self.writer.add_image(f"test/original_{i}", original_tensor, idx)
                        self.writer.add_image(f"test/ground_truth_{i}", gt_tensor, idx)
                        self.writer.add_image(f"test/prediction_{i}", pred_tensor, idx)
                        self.writer.add_image(f"test/prediction_rain_{i}", pred_tensor_rain, idx)
                        self.writer.add_image(f"test/ground_truth_rain_{i}", gt_img_rain, idx)
                        # Add debugging information

            pred = pred.max(1)[1].data.cpu().numpy()
            # print(f"pred: {pred}")

            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                tn = data_dict["origin_tn"]

                final_scan_mask = tn.squeeze(1) == 0

                np.save(pred_save_path, pred)
            
            pred_labels = pred[final_scan_mask]
            gt_labels = segment[final_scan_mask]
            TP = np.sum((pred_labels == 0) & (gt_labels == 0))
            TN = np.sum((pred_labels == 1) & (gt_labels == 1))
            GP = np.sum(gt_labels == 0)
            GN = np.sum(gt_labels == 1)
            
            SR = TP / GP if GP > 0 else 0
            NR = TN / GN if GN > 0 else 0
            DA = 0.5 * (SR + NR)
            
            intersection, union, target = intersection_and_union(
                pred[final_scan_mask], segment[final_scan_mask], self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target, SR=SR, NR=NR, DA=DA
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            if mask[1]:
                moving_meter.update(iou_class[1])
            moving_iou = iou_class[1] if mask[1] else 1
            #TODO
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            total = intersection_meter.sum / (union_meter.sum + 1e-10)
            moving_m_iou = total[1]
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f}) "
                "movingIoU {moving_iou:.4f} ({moving_m_iou:.4f}) "
                "SR:{SR:.4f} || NR:{NR:.4f} || DA:{DA:.4f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                    moving_iou=moving_iou,
                    moving_m_iou=moving_m_iou,
                    SR=SR,
                    NR=NR,
                    DA=DA,
                )
            )
            if self.cfg.data.test.type == "EventRainMultiScansDataset":
                # 00_000000 -> 00, 000000
                # print(f"Data name: {data_name}")
                sequence_name, frame_name = data_name.rsplit("_", 1)
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred[final_scan_mask].astype(np.int32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.int32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "SPACMultiScansDataset" or self.cfg.data.test.type == "SPACMultiScansDataset_V2":
                # 00_000000 -> 00, 000000
                sequence_name = data_name.split("+")[0]
                frame_name = data_name.split("+")[1]
                print(f"Data name: {data_name} || sequence_name name: {sequence_name} || frame_name name: {frame_name}")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred[final_scan_mask].astype(np.int32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.int32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            SR = np.mean([meters["SR"] for _, meters in record.items()], axis=0)
            NR = np.mean([meters["NR"] for _, meters in record.items()], axis=0)
            DA = np.mean([meters["DA"] for _, meters in record.items()], axis=0)
            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("SR:{:.4f} || NR:{:.4f} || DA:{:.4f}".format(SR,NR,DA))
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class SemSegTester_EVK4(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False):
        super().__init__(cfg, model=model, test_loader=test_loader, verbose=verbose)
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.save_path, 'testlogs'))
        self.shape = cfg.size
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        moving_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                (self.cfg.data.test.type == "SemanticKITTIDataset" or
                 self.cfg.data.test.type == "SemanticKITTIMultiScansDataset") and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            tn = data_dict.pop("tn")
            final_scan_mask = tn.squeeze(1) == 0

            pred_save_path = os.path.join(save_path, "{}.npy".format(data_name))

            # data_name format: scene{x}/rain_{y}/{z}
            scene, seq, frame = data_name.split('/')
            scene_save_path = os.path.join(save_path, scene)
            seq_save_path = os.path.join(scene_save_path, seq)
            make_dirs(seq_save_path)
            pred_save_path = os.path.join(seq_save_path, f"{frame}_pred.npy")


            pred_count = torch.zeros(segment.size).cuda() 
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            coord = torch.zeros((segment.size, 3)).cuda()
            p = torch.zeros((segment.size, 1)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    pred_part = self.model(input_dict)["seg_logits"]
                    torch.cuda.synchronize() 
                    inference_time = time.perf_counter() - start_time
                    
                    if idx > 0: 
                        self.writer.add_scalar("inference_time", inference_time * 1000, idx)

                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    coord_part = input_dict["coord"]
                    p_part = input_dict["feat"][:, 3].unsqueeze(1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        pred_count[idx_part[bs:be]] += 1
                        coord[idx_part[bs:be], :] = coord_part[bs:be]
                        p[idx_part[bs:be], :] = p_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i+1,
                        batch_num=len(fragment_list),
                    )
                )

                if i == 0:
                    coord_np = coord.cpu().numpy()
                    pred_np = pred.cpu().numpy() 
                    p_np = p.cpu().numpy()
                    p_np = p.squeeze()

                    time_stamps = coord_np[:, 2]
                    min_time = np.min(time_stamps)
                    time_window = 0.45
                    target_time = min_time + time_window
                    time_mask = time_stamps <= target_time

                    x = coord_np[time_mask, 0]
                    y = coord_np[time_mask, 1]
                    p_vis = p_np[time_mask]
                    pred_vis = pred_np[time_mask]
                    segment = segment[time_mask]

                    p_vis = p_vis.cpu().numpy() if isinstance(p_vis, torch.Tensor) else p_vis

                    def render(x, y, p, mask, shape):
                        img = np.full(shape=tuple(shape) + (3,), fill_value=255, dtype="uint8") 
                        valid = mask
                        x, y, p = x[valid], y[valid], p[valid]
                        x = x.astype(int)
                        y = y.astype(int)
                        img[y, x, :] = 0 

                        def hex_to_rgb(hex_color):
                            hex_color = hex_color.lstrip('#')
                            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        import seaborn as sns
                        pos_color = hex_to_rgb(sns.xkcd_rgb['dark pink'])  # Positive events color
                        neg_color = hex_to_rgb(sns.xkcd_rgb['sky blue'])  # Negative events color

                        # Apply colors
                        pos_mask = (p > 0)
                        neg_mask = (p < 0)
                        img[y[pos_mask], x[pos_mask]] = pos_color
                        img[y[neg_mask], x[neg_mask]] = neg_color
                        return img

                    p_np = p_np if isinstance(p_np, np.ndarray) else p_np.cpu().numpy()
                    original_img = render(x, y, p_vis, np.ones_like(p_vis, dtype=bool), self.shape)

                    # Generate ground truth image (label == 1 only)
                    gt_mask = segment == 1
                    gt_img = render(x, y, p_vis, gt_mask, self.shape)

                    gt_mask_rain = segment == 0
                    gt_img_rain = render(x, y, p_vis, gt_mask_rain, self.shape)

                    max_indices = np.argmax(pred_vis, axis=1)  

                    # Generate predicted image (label == 1 only)
                    pred_mask = max_indices == 1
                    pred_img = render(x, y, p_vis, pred_mask, self.shape)

                    # Generate predicted image (label == 0 only)
                    pred_mask_rain = max_indices == 0 
                    pred_img_rain = render(x, y, p_vis, pred_mask_rain, self.shape)

                    original_tensor = torch.tensor(original_img).permute(2, 0, 1) / 255.0 
                    gt_tensor = torch.tensor(gt_img).permute(2, 0, 1) / 255.0
                    pred_tensor = torch.tensor(pred_img).permute(2, 0, 1) / 255.0
                    pred_tensor_rain = torch.tensor(pred_img_rain).permute(2, 0, 1) / 255.0
                    gt_img_rain = torch.tensor(gt_img_rain).permute(2, 0, 1) / 255.0

                    if self.writer is not None:
                        self.writer.add_image(f"test/original_{i}", original_tensor, idx)
                        self.writer.add_image(f"test/ground_truth_{i}", gt_tensor, idx)
                        self.writer.add_image(f"test/prediction_{i}", pred_tensor, idx)
                        self.writer.add_image(f"test/prediction_rain_{i}", pred_tensor_rain, idx)
                        self.writer.add_image(f"test/ground_truth_rain_{i}", gt_img_rain, idx)
                        self.writer.flush() 

            pred = pred / pred_count.unsqueeze(1)

            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
                tn = data_dict["origin_tn"]

                final_scan_mask = tn.squeeze(1) == 0
            assert len(pred) == len(segment), "Predicted points and labeled points mismatch"

            pred = pred.max(1)[1].data.cpu().numpy()

            np.save(pred_save_path, pred)

            pred_labels = pred[final_scan_mask]
            gt_labels = segment[final_scan_mask]
            TP = np.sum((pred_labels == 0) & (gt_labels == 0))
            TN = np.sum((pred_labels == 1) & (gt_labels == 1))
            GP = np.sum(gt_labels == 0)
            GN = np.sum(gt_labels == 1)
            SR = TP / GP if GP > 0 else 0
            NR = TN / GN if GN > 0 else 0
            DA = 0.5 * (SR + NR)


            intersection, union, target = intersection_and_union(
                pred[final_scan_mask], segment[final_scan_mask], self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target, SR=SR, NR=NR, DA=DA
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            if mask[1]:
                moving_meter.update(iou_class[1])
            moving_iou = iou_class[1] if mask[1] else 1
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            total = intersection_meter.sum / (union_meter.sum + 1e-10)
            moving_m_iou = total[1]
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f}) "
                "movingIoU {moving_iou:.4f} ({moving_m_iou:.4f}) "
                "SR:{SR:.4f} || NR:{NR:.4f} || DA:{DA:.4f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                    moving_iou=moving_iou,
                    moving_m_iou=moving_m_iou,
                    SR=SR,
                    NR=NR,
                    DA=DA,
                )
            )
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                scene_name, seq_name, frame_name = data_name.split('/')
                save_dir = os.path.join(
                    save_path, 
                    "submit", 
                    "sequences",
                    f"{scene_name}", 
                    f"{seq_name}",
                    "predictions"
                )
                os.makedirs(save_dir, exist_ok=True)

                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_dir,
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "SemanticKITTIMultiScansDataset":
                scene_name, seq_name, frame_name = data_name.split('/')
                save_dir = os.path.join(
                    save_path, 
                    "submit", 
                    "sequences",
                    f"{scene_name}", 
                    f"{seq_name}",
                    "predictions"
                )
                os.makedirs(save_dir, exist_ok=True)

                pred = pred[final_scan_mask].astype(np.int32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.int32)
                pred.tofile(
                    os.path.join(
                        save_dir,
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            length = len(self.test_loader)
            SR = np.sum([meters["SR"] for _, meters in record.items()], axis=0)/length
            NR = np.sum([meters["NR"] for _, meters in record.items()], axis=0)/length
            DA = np.sum([meters["DA"] for _, meters in record.items()], axis=0)/length

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("SR:{:.4f} || NR:{:.4f} || DA:{:.4f}".format(SR,NR,DA))
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)



@TESTERS.register_module()
class PointDenoiseTester(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False):
        super().__init__(cfg, model=model, test_loader=test_loader, verbose=verbose)
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.save_path, 'testlogs'))
    
    def render(self, x, y, t, p, shape):
        # Render a 2D image from the point cloud coordinates (x, y) and labels (p)
        p = (p > 0).astype(int)
        img = np.full(shape=tuple(shape) + (3,), fill_value=255, dtype="uint8")
        img[y, x, :] = 0
        img[y, x, p] = 255
        return img

    def visualize_points(self, points, tag, idx):
        # Extract x, y, z, and p from the points
        x, y, _, p = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        
        # Normalize x and y to the range [0, 460] and [0, 352]
        # x_min, x_max = x.min(), x.max()
        # y_min, y_max = y.min(), y.max()

        x = (np.clip(x, 0, 1) * 460).astype(int)
        y = (np.clip(y, 0, 1) * 352).astype(int)

        valid_mask = (x >= 0) & (x < 460) & (y >= 0) & (y < 352)
        x = x[valid_mask]
        y = y[valid_mask]
        p = p[valid_mask]
        
        # Render the points into an image
        img = self.render(x, y, None, p, (352, 460))

        img = img.astype(np.uint8)
        img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W) format for TensorBoard

        # Add image to the TensorBoard writer
        if self.writer is not None:
            self.writer.add_image(tag, img, idx)

    def compute_metrics(self, denoised_points, clean_points):
        # Use KDTree for nearest neighbor search
        # print(f"Denoised points shape: {denoised_points.shape}")
        # print(f"Clean points shape: {clean_points.shape}")
        tree = cKDTree(clean_points[:, :3])
        dist, idx = tree.query(denoised_points[:, :3])

        # Compute MSE
        mse = np.mean(dist ** 2)

        # Compute PSNR
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        return {"mse": mse.item(), "psnr": psnr.item()}
    '''
    def compute_metrics(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        mse = F.mse_loss(pred, target)

        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean((pred - target) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        return {"mse": mse.item(), "snr": snr.item()}
    '''

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        mse_meter = AverageMeter()
        psnr_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        record = {}

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # Currently, assume the batch size is 1.
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            tn = data_dict.pop("tn")
            final_scan_mask = tn.squeeze(1) == 0

            pred_save_path = os.path.join(save_path, "{}.npy".format(data_name))

            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()

            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    start_forward_time = time.time()
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    end_forward_time = time.time()
                    forward_time = end_forward_time - start_forward_time
                    print(f"Forward time: {forward_time* 1000}")
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i+1,
                        batch_num=len(fragment_list),
                    )
                )
            pred = pred.data.cpu().numpy()

            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                # segment = data_dict["origin_segment"]
                tn = data_dict["origin_tn"]


            # Calculate the denoising metrics
            metrics = self.compute_metrics(pred, segment)
            mse_meter.update(metrics["mse"])
            psnr_meter.update(metrics["psnr"])

            logger.info(
                "Test: {}/{}-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "MSE {mse:.4f} ({mse_avg:.4f}) "
                "PSNR {psnr:.2f} ({psnr_avg:.2f}) ".format(
                    idx + 1,
                    len(self.test_loader),
                    data_name,
                    batch_time=batch_time,
                    mse=metrics["mse"],
                    mse_avg=mse_meter.avg,
                    psnr=metrics["psnr"],
                    psnr_avg=psnr_meter.avg,
                )
            )
            # Log MSE and PSNR to TensorBoard
            self.writer.add_scalar(f"Test/{data_name}/MSE", metrics["mse"], idx)
            self.writer.add_scalar(f"Test/{data_name}/PSNR", metrics["psnr"], idx)
            
            # Visualize and log the points (original, denoised, and clean)
            original_points = input_dict["feat"][:, :4].cpu().numpy()
            # print(f"Original points: {original_points.shape}")
            # print(f"Pred points: {pred.shape}")
            # print(f"Segment points: {segment.shape}")
            self.visualize_points(original_points, "original",idx)
            self.visualize_points(pred, "pred",idx)
            self.visualize_points(segment, "segment",idx)


            # Save prediction results
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")
            np.save(pred_save_path, pred)

            # segment_save_path = os.path.join(save_path, f"{data_name}_segment.npy")
            # np.save(segment_save_path, segment)
            record[data_name] = metrics

            batch_time.update(time.time() - end)
            # break

        logger.info("Syncing ...")
        comm.synchronize()

        # Print the final result
        logger.info(
            "Final result: "
            "MSE: {:.4f}, SNR: {:.2f}".format(
                mse_meter.avg, psnr_meter.avg
            )
        )

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        return record
    
    @staticmethod
    def collate_fn(batch):
        return batch
