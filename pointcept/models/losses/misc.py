
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .builder import LOSSES
# from pytorch3d.ops import knn_points, knn_gather
import faiss
import numpy as np

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class FrequencyLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0, ignore_index=-1):
        super(FrequencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred_labels = torch.argmax(pred, dim=1) # (N, C) -> (N,)

        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred_labels = pred_labels[mask].contiguous()
            target = target[mask].contiguous()

        pred_labels = pred_labels.float().view(-1, 1)
        target = target.float().view(-1, 1)

        with torch.no_grad():
            pred_fft = torch.fft.fft(pred_labels).abs()
            target_fft = torch.fft.fft(target).abs()

        diff = torch.abs(pred_fft - target_fft)
        diff = diff / (torch.max(diff, torch.tensor(1e-6, device=diff.device)) + 1e-8)
        loss = torch.mean(diff**2)

        # reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_weight 

@LOSSES.register_module()
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        self.loss = nn.CrossEntropyLoss(
            weight=None,  
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        # Dynamic Weight Calculation
        class_counts = torch.bincount(target.flatten(), minlength=pred.size(1))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights /= class_weights.sum()  # 归一化权重
        class_weights = class_weights.to(pred.device)

        # Update weights to the loss
        self.loss.weight = class_weights
        loss = self.loss(pred, target) * self.loss_weight
        # print(f"Class weights: {class_weights.cpu().numpy()}")
        # print(f"Loss: {loss.item()}")
        # Calculate loss and weight
        return loss

@LOSSES.register_module()
class BCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        pos_weight=None,  # Used for imbalanced datasets
    ):
        super(BCEWithLogitsLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        pos_weight = torch.tensor(pos_weight).cuda() if pos_weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.BCEWithLogitsLoss(
            weight=weight,  # Weights applied to each sample
            reduction=reduction,  
            pos_weight=pos_weight,  # Weights of positive samples (for category imbalance)
        )

    def forward(self, pred, target):
        target = target.float()  
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        target = target[:, None].float()
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        # print("Prediction Tensor: ", pred)
        # print("Target Tensor: ", target)
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss

@LOSSES.register_module()
class CombinedPointCloudLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, intensity_weight=0.5, use_gpu=True):
        # A class for calculating the loss of merged point clouds, supporting Chamfer Distance and Intensity Loss.
        super(CombinedPointCloudLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.intensity_weight = intensity_weight
        self.use_gpu = use_gpu

    def knn_search_batch(self, pred_coords, target_coords, batch_size=10000, use_gpu=True):
        """
        Use Faiss for nearest neighbor search in batches to reduce memory pressure.
        """
        if use_gpu:
            res = faiss.StandardGpuResources()  
            index = faiss.IndexFlatL2(3)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  
        else:
            index = faiss.IndexFlatL2(3)

        # Build index
        if use_gpu:
            gpu_index.add(target_coords)
        else:
            index.add(target_coords)

        # Batch processing forecast points
        num_pred = pred_coords.shape[0]
        indices_pred_to_target = []

        for i in range(0, num_pred, batch_size):
            batch_pred = pred_coords[i:i + batch_size]
            if use_gpu:
                _, indices = gpu_index.search(batch_pred, 1)
            else:
                _, indices = index.search(batch_pred, 1)
            indices_pred_to_target.append(indices)

        indices_pred_to_target = np.concatenate(indices_pred_to_target, axis=0).squeeze(1)

        # Build reverse index (target -> prediction)
        if use_gpu:
            gpu_index.reset()
            gpu_index.add(pred_coords)
            _, indices_target_to_pred = gpu_index.search(target_coords, 1)
        else:
            index.reset()
            index.add(pred_coords)
            _, indices_target_to_pred = index.search(target_coords, 1)

        return indices_pred_to_target, indices_target_to_pred.squeeze(1)


    def forward(self, pred, target):
        """
        Forward calculation of the merging loss of two point clouds.
        """
        # 提取坐标和强度
        pred_coords = pred[:, :3].detach().cpu().numpy()  # (N, 3)
        target_coords = target[:, :3].detach().cpu().numpy()  # (M, 3)
        pred_intensity = pred[:, 3]  # (N,)
        target_intensity = target[:, 3]  # (M,)

        # Recent Nearest Search
        indices_pred_to_target, indices_target_to_pred = self.knn_search_batch(pred_coords, target_coords)  # Pred -> Target

        chamfer_loss = 0
        if self.chamfer_weight > 0:
            dist_pred_to_target = torch.norm(pred[:, :3] - target[indices_pred_to_target, :3], dim=1)
            dist_target_to_pred = torch.norm(target[:, :3] - pred[indices_target_to_pred, :3], dim=1)
            chamfer_loss = torch.mean(dist_pred_to_target) + torch.mean(dist_target_to_pred)

        intensity_loss = 0
        if self.intensity_weight > 0:
            matched_target_intensity = target_intensity[indices_pred_to_target]
            intensity_loss = F.mse_loss(pred_intensity, matched_target_intensity)

        total_loss = self.chamfer_weight * chamfer_loss + self.intensity_weight * intensity_loss
        return total_loss

@LOSSES.register_module()
class ChamferDistance(nn.Module):
    def __init__(self, loss_weight=1.0, use_gpu=True):
        """
        Chamfer Distance loss, used to measure the distance between point clouds.
        """
        super(ChamferDistance, self).__init__()
        self.loss_weight = loss_weight
        self.use_gpu = use_gpu

    def forward(self, adv_pc, ori_pc):
        adv_xyz = adv_pc[:, :3].detach().cpu().numpy()   # (N, 3)
        ori_xyz = ori_pc[:, :3].detach().cpu().numpy()  # (M, 3)

        # Use Faiss for nearest neighbor search
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Create GPU resources
            index = faiss.IndexFlatL2(3)  # L2 Distance Index
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Migrate the index to the GPU
            gpu_index.add(ori_xyz)  # Add the target point cloud to the index.

            _, indices_adv_to_ori = gpu_index.search(adv_xyz, 1)  
            _, indices_ori_to_adv = gpu_index.search(ori_xyz, 1)  
        else:
            index = faiss.IndexFlatL2(3)  
            index.add(ori_xyz) 
            _, indices_adv_to_ori = index.search(adv_xyz, 1)  
            _, indices_ori_to_adv = index.search(ori_xyz, 1)  

        min_dist_adv_to_ori = torch.mean(torch.from_numpy(indices_adv_to_ori).float())  
        min_dist_ori_to_adv = torch.mean(torch.from_numpy(indices_ori_to_adv).float()) 

        chamfer_loss = min_dist_adv_to_ori + min_dist_ori_to_adv
        return chamfer_loss * self.loss_weight


@LOSSES.register_module()
class IntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_gpu=True):
        super(IntensityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_gpu = use_gpu

    def forward(self, pred, target):
        
        pred_intensity = pred[:, 3]  # Predicted intensity
        target_intensity = target[:, 3]  # GT Strength

        # Extract the spatial coordinates of the points
        pred_coords = pred[:, :3].detach().cpu().numpy()  # (N, 3)
        target_coords = target[:, :3].detach().cpu().numpy()  # (M, 3)

        # Build Faiss index
        if self.use_gpu:
            res = faiss.StandardGpuResources()  
            index = faiss.IndexFlatL2(3)  
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  
            gpu_index.add(target_coords)  
            _, indices = gpu_index.search(pred_coords, 1)  
        else:
            index = faiss.IndexFlatL2(3)  
            index.add(target_coords) 
            _, indices = index.search(pred_coords, 1)  

        indices = indices.squeeze(1)  # (N,)
        matched_target_intensity = target_intensity[indices]

        intensity_loss = F.mse_loss(pred_intensity, matched_target_intensity)
        return intensity_loss * self.loss_weight

@LOSSES.register_module()
class ChamferDistanceWithPolarity(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, beta=0.5):
        """
        Chamfer Distance loss measures the distance between point clouds by combining polarity information.
        """
        super(ChamferDistanceWithPolarity, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha 
        self.beta = beta    

    def forward(self, pred, target):
        pred_xyz = pred[:, :3]
        target_xyz = target[:, :3]  
        pred_p = pred[:, 3].unsqueeze(1)  
        target_p = target[:, 3].unsqueeze(0)  

        dist_xyz = torch.cdist(pred_xyz, target_xyz, p=2) ** 2  

        dist_p = (pred_p - target_p) ** 2 

        combined_dist = self.alpha * dist_xyz + self.beta * dist_p  # (N, M)

        min_dist_pred_to_gt = combined_dist.min(dim=1)[0]  
        min_dist_gt_to_pred = combined_dist.min(dim=0)[0] 

        chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

        return chamfer_loss * self.loss_weight

# @LOSSES.register_module()
# class chamfer_loss_with_intensity(nn.Module):
#     def __init__(self, loss_weight=1.0, intensity_weight=0.5):
#         super(chamfer_loss_with_intensity, self).__init__()
#         self.loss_weight = loss_weight
#         self.intensity_weight = intensity_weight

#     def forward(self, adv_pc, ori_pc):
#         device = adv_pc.device

#         adv_xyz = adv_pc[:, :3]
#         ori_xyz = ori_pc[:, :3]

#         adv_KNN = knn_points(adv_xyz.unsqueeze(0), ori_xyz.unsqueeze(0), K=1)
#         ori_KNN = knn_points(ori_xyz.unsqueeze(0), adv_xyz.unsqueeze(0), K=1)

#         min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
#         min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
#         chamfer_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)

#         adv_intensity = adv_pc[:, 3]
#         ori_intensity = ori_pc[:, 3]

#         adv_indices = adv_KNN.idx.squeeze(0).squeeze(-1).to(device).long()
#         ori_indices = ori_KNN.idx.squeeze(0).squeeze(-1).to(device).long()

#         ori_intensity_aligned = ori_intensity[adv_indices]
#         adv_intensity_diff_1 = torch.mean((adv_intensity - ori_intensity_aligned) ** 2)

#         adv_intensity_aligned = adv_intensity[ori_indices]
#         adv_intensity_diff_2 = torch.mean((ori_intensity - adv_intensity_aligned) ** 2)

#         intensity_loss = (adv_intensity_diff_1 + adv_intensity_diff_2) / 2
#         total_loss = chamfer_loss * self.loss_weight + intensity_loss * self.intensity_weight
#         return total_loss

@LOSSES.register_module()
class AsymmetricHausdorffDistance(nn.Module):
    def __init__(self, loss_weight=1.0):

        super(AsymmetricHausdorffDistance, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):

        pred = pred[:, :3]
        target = target[:, :3]

        dist = torch.cdist(pred, target, p=2)  # (N, M)

        min_dist_pred_to_target = dist.min(dim=1)[0]  


        hausdorff_loss = min_dist_pred_to_target.max()  
        return hausdorff_loss * self.loss_weight

# @LOSSES.register_module()
# class CurvatureLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super(CurvatureLoss, self).__init__()
#         self.loss_weight = loss_weight

#     def forward(self, adv_pc, ori_pc):

#         adv_pc = adv_pc[:, :3]  
#         ori_pc = ori_pc[:, :3] 

#         kappa_adv, _ = self._get_kappa_adv(adv_pc, ori_pc, k=2)


#         kappa_ori, _ = self._get_kappa_adv(ori_pc, ori_pc, k=2)

#         knn_result = knn_points(adv_pc.unsqueeze(0), ori_pc.unsqueeze(0), K=1)
#         _, indices, _ = knn_result.dists, knn_result.idx, knn_result.knn
#         kappa_ori_matched = kappa_ori[indices.squeeze(0).squeeze(1)]  # (N,)

#         curv_loss = ((kappa_adv - kappa_ori_matched) ** 2).mean()

#         return curv_loss * self.loss_weight

#     def _get_kappa_adv(self, adv_pc, ori_pc, k=10):


#         adv_pc = adv_pc.unsqueeze(0) 
#         ori_pc = ori_pc.unsqueeze(0)  

#         inter_KNN = knn_points(adv_pc, adv_pc, K=k)
#         nn_pts = knn_gather(adv_pc, inter_KNN.idx).squeeze(0)[:, 1:, :]  # (N, k, 3)

#         vectors = nn_pts - adv_pc.squeeze(0).unsqueeze(1)  # (N, k, 3)
#         cov_matrices = torch.matmul(vectors.transpose(1, 2), vectors) / k  # (N, 3, 3)
#         eigvals, _ = torch.linalg.eigh(cov_matrices)  # (N, 3)

#         kappa_adv = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-6)  # (N,)

#         return kappa_adv, None  

