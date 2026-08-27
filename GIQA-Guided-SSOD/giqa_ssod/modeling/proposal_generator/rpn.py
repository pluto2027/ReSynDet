# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified to support per-image weighting of RPN losses.
from typing import Dict, List, Optional
import torch

from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator import RPN
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY


@PROPOSAL_GENERATOR_REGISTRY.register()
class PseudoLabRPN(RPN):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        iqa_weights: Optional[List[float]] = None,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(
                x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if (self.training and compute_loss) or compute_val_loss:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)

            use_iqa_weights = False
            if iqa_weights is not None:
                use_iqa_weights = any(float(w) != 1.0 for w in iqa_weights)

            if not use_iqa_weights:
                losses = self.losses(
                    anchors,
                    pred_objectness_logits,
                    gt_labels,
                    pred_anchor_deltas,
                    gt_boxes,
                )
            else:
                loss_sums = {}
                num_images = max(1, len(gt_labels))

                for img_idx in range(len(gt_labels)):
                    per_obj_logits = [
                        score[img_idx].unsqueeze(0)
                        for score in pred_objectness_logits
                    ]
                    per_anchor_deltas = [
                        delta[img_idx].unsqueeze(0)
                        for delta in pred_anchor_deltas
                    ]
                    losses_i = self.losses(
                        anchors,
                        per_obj_logits,
                        [gt_labels[img_idx]],
                        per_anchor_deltas,
                        [gt_boxes[img_idx]],
                    )
                    weight = float(iqa_weights[img_idx])
                    for key, value in losses_i.items():
                        if key not in loss_sums:
                            loss_sums[key] = value * weight
                        else:
                            loss_sums[key] = loss_sums[key] + value * weight

                losses = {key: value / num_images for key, value in loss_sums.items()}

            losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        else:  # inference
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )

        return proposals, losses
