# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified to load GIQA scores and compute per-image weights.
import copy
import csv
import logging
import os
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from giqa_ssod.data.detection_utils import build_strong_augmentation


class DatasetMapperTwoCropSeparate(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        # IQA-based per-image weighting (used for unlabeled data)
        self.iqa_weight_on = bool(getattr(cfg.SEMISUPNET, "IQA_WEIGHT_ON", False))
        self.iqa_score_path = getattr(cfg.SEMISUPNET, "IQA_SCORE_PATH", "")
        self.iqa_score_field = getattr(cfg.SEMISUPNET, "IQA_SCORE_FIELD", "norm_score")
        self.iqa_score_scale = float(getattr(cfg.SEMISUPNET, "IQA_SCORE_SCALE", 100.0))
        self.iqa_weight_mapping = getattr(
            cfg.SEMISUPNET, "IQA_WEIGHT_MAPPING", "linear"
        ).lower()
        self.iqa_weight_min = float(getattr(cfg.SEMISUPNET, "IQA_WEIGHT_MIN", 0.0))
        self.iqa_weight_max = float(getattr(cfg.SEMISUPNET, "IQA_WEIGHT_MAX", 1.0))
        self.iqa_weight_normalize_mean = bool(
            getattr(cfg.SEMISUPNET, "IQA_WEIGHT_NORMALIZE_MEAN", False)
        )
        self.iqa_weight_target_mean = float(
            getattr(cfg.SEMISUPNET, "IQA_WEIGHT_TARGET_MEAN", 1.0)
        )
        self.iqa_filter_on = bool(getattr(cfg.SEMISUPNET, "IQA_FILTER_ON", False))
        self.iqa_filter_threshold = float(
            getattr(cfg.SEMISUPNET, "IQA_FILTER_THRESHOLD", 0.0)
        )
        self.iqa_filter_missing_score = bool(
            getattr(cfg.SEMISUPNET, "IQA_FILTER_MISSING_SCORE", False)
        )
        self.iqa_weight_normalize_factor = 1.0
        self._iqa_score_map = {}
        if self.iqa_weight_on:
            if self.iqa_score_path:
                self._iqa_score_map = self._load_iqa_scores(
                    self.iqa_score_path, self.iqa_score_field
                )
            else:
                logging.getLogger(__name__).warning(
                    "IQA_WEIGHT_ON is True but IQA_SCORE_PATH is empty."
                )

    def _load_iqa_scores(self, path: str, score_field: str) -> dict:
        score_map = {}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if score_field not in reader.fieldnames:
                raise ValueError(
                    "IQA score field '{}' not found in {}".format(score_field, path)
                )
            for row in reader:
                file_name = row.get("filename")
                if not file_name:
                    continue
                key = os.path.basename(file_name)
                try:
                    score = float(row[score_field])
                except ValueError:
                    continue
                score_map[key] = score
        return score_map

    def _normalize_iqa_score(self, score: float) -> float:
        s = float(score)
        if self.iqa_score_scale not in (0.0, 1.0):
            s = s / self.iqa_score_scale
        return s

    def _clip_iqa_weight(self, weight: float) -> float:
        return float(max(self.iqa_weight_min, min(self.iqa_weight_max, weight)))

    def _compute_iqa_weight(self, score: float) -> float:
        if score is None:
            return 1.0

        s = self._normalize_iqa_score(score)
        if self.iqa_weight_mapping != "linear":
            raise ValueError(
                "Unsupported IQA_WEIGHT_MAPPING '{}'; this implementation uses linear weighting.".format(
                    self.iqa_weight_mapping
                )
            )

        weight = self.iqa_weight_min + (
            self.iqa_weight_max - self.iqa_weight_min
        ) * s

        return self._clip_iqa_weight(weight)

    def _get_raw_iqa_weight(self, file_name: str) -> float:
        key = os.path.basename(file_name)
        score = self._iqa_score_map.get(key)
        return self._compute_iqa_weight(score)

    def _get_iqa_score(self, file_name: str):
        key = os.path.basename(file_name)
        return self._iqa_score_map.get(key)

    def _keep_by_iqa_score(self, file_name: str) -> bool:
        if not self.iqa_weight_on or not self.iqa_filter_on:
            return True

        score = self._get_iqa_score(file_name)
        if score is None:
            return not self.iqa_filter_missing_score

        return self._normalize_iqa_score(score) >= self.iqa_filter_threshold

    def filter_unlabeled_dataset_by_iqa(self, dataset_dicts):
        if not self.iqa_weight_on or not self.iqa_filter_on:
            return dataset_dicts

        kept = [
            d for d in dataset_dicts
            if "file_name" in d and self._keep_by_iqa_score(d["file_name"])
        ]
        removed = len(dataset_dicts) - len(kept)
        logging.getLogger(__name__).info(
            "IQA filtering: kept={} removed={} threshold={:.6f} missing_score_filtered={}".format(
                len(kept),
                removed,
                self.iqa_filter_threshold,
                self.iqa_filter_missing_score,
            )
        )
        if len(kept) == 0:
            raise ValueError(
                "IQA filtering removed all unlabeled images. Lower SEMISUPNET.IQA_FILTER_THRESHOLD or disable IQA_FILTER_ON."
            )
        return kept

    def set_iqa_normalization_from_dataset(self, dataset_dicts):
        if not self.iqa_weight_on or not self.iqa_weight_normalize_mean:
            self.iqa_weight_normalize_factor = 1.0
            return

        weights = [
            self._get_raw_iqa_weight(d["file_name"])
            for d in dataset_dicts
            if "file_name" in d
        ]
        if len(weights) == 0:
            self.iqa_weight_normalize_factor = 1.0
            logging.getLogger(__name__).warning(
                "IQA mean normalization is enabled but no unlabeled images were found."
            )
            return

        raw_mean = float(np.mean(weights))
        if raw_mean <= 0:
            self.iqa_weight_normalize_factor = 1.0
            logging.getLogger(__name__).warning(
                "IQA mean normalization is skipped because raw mean is {:.6f}.".format(
                    raw_mean
                )
            )
            return

        self.iqa_weight_normalize_factor = self.iqa_weight_target_mean / raw_mean
        logging.getLogger(__name__).info(
            "IQA mean normalization: n={} raw_mean={:.6f} target_mean={:.6f} factor={:.6f}".format(
                len(weights),
                raw_mean,
                self.iqa_weight_target_mean,
                self.iqa_weight_normalize_factor,
            )
        )

    def _get_iqa_weight(self, file_name: str) -> float:
        if not self.iqa_weight_on:
            return 1.0
        return self._get_raw_iqa_weight(file_name) * self.iqa_weight_normalize_factor

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        if dataset_dict.get("is_unlabeled", False):
            dataset_dict["iqa_weight"] = self._get_iqa_weight(
                dataset_dict["file_name"]
            )
        else:
            dataset_dict["iqa_weight"] = 1.0

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)
