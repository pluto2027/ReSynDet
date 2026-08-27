#!/usr/bin/env python3
"""Train GIQA-Guided SSOD on COCO-format labeled and unlabeled datasets."""

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import default_argument_parser, default_setup, launch

from giqa_ssod import add_giqa_ssod_config
from giqa_ssod.data.datasets.builtin import load_coco_unlabel_json
from giqa_ssod.engine.trainer import BaselineTrainer, GIQAGuidedTeacherTrainer
from giqa_ssod.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN  # noqa: F401
from giqa_ssod.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from giqa_ssod.modeling.proposal_generator.rpn import PseudoLabRPN  # noqa: F401
from giqa_ssod.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab  # noqa: F401


LABEL_DATASET = "giqa_train_label"
UNLABEL_DATASET = "giqa_train_unlabel"
VAL_DATASET = "giqa_val"


def _replace_dataset(name, loader, json_file, image_root, class_names):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(
        thing_classes=class_names,
        evaluator_type="coco",
        json_file=json_file,
        image_root=image_root,
    )


def register_datasets(args):
    if args.eval_only:
        required = (args.val_json, args.val_root)
        if not all(required):
            raise ValueError("--val-json and --val-root are required for evaluation.")
    else:
        required = (
            args.labeled_json,
            args.labeled_root,
            args.unlabeled_json,
            args.unlabeled_root,
            args.val_json,
            args.val_root,
        )
        if not all(required):
            raise ValueError(
                "Training requires labeled, unlabeled, and validation JSON/root arguments."
            )

    class_names = list(args.class_names)
    if args.labeled_json and args.labeled_root:
        _replace_dataset(
            LABEL_DATASET,
            lambda: load_coco_json(
                args.labeled_json, args.labeled_root, LABEL_DATASET
            ),
            args.labeled_json,
            args.labeled_root,
            class_names,
        )
    if args.unlabeled_json and args.unlabeled_root:
        _replace_dataset(
            UNLABEL_DATASET,
            lambda: load_coco_unlabel_json(
                args.unlabeled_json, args.unlabeled_root, UNLABEL_DATASET
            ),
            args.unlabeled_json,
            args.unlabeled_root,
            class_names,
        )
    _replace_dataset(
        VAL_DATASET,
        lambda: load_coco_json(args.val_json, args.val_root, VAL_DATASET),
        args.val_json,
        args.val_root,
        class_names,
    )


def setup(args):
    cfg = get_cfg()
    add_giqa_ssod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_datasets(args)

    if cfg.SEMISUPNET.Trainer == "giqa_ssod":
        trainer_class = GIQAGuidedTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        trainer_class = BaselineTrainer
    else:
        raise ValueError(
            "Unknown SEMISUPNET.Trainer: {}".format(cfg.SEMISUPNET.Trainer)
        )

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "giqa_ssod":
            student = trainer_class.build_model(cfg)
            teacher = trainer_class.build_model(cfg)
            ensemble = EnsembleTSModel(teacher, student)
            DetectionCheckpointer(ensemble, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            return trainer_class.test(cfg, ensemble.modelTeacher)

        model = trainer_class.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return trainer_class.test(cfg, model)

    trainer = trainer_class(cfg)
    if args.resume:
        # A resumed checkpoint contains the complete teacher-student ensemble,
        # optimizer, scheduler, and iteration state.
        trainer.resume_or_load(resume=True)
    else:
        # MODEL.WEIGHTS is a plain Detectron2 detector checkpoint in the plant
        # experiments. Loading it through the ensemble checkpointer would try
        # to match plain "backbone.*" keys against "modelTeacher.*" and
        # "modelStudent.*", leaving the detector effectively uninitialized.
        DetectionCheckpointer(
            trainer.model, save_dir=cfg.OUTPUT_DIR
        ).load(cfg.MODEL.WEIGHTS)
        if hasattr(trainer, "model_teacher"):
            trainer._copy_main_model()
        trainer.start_iter = 0
    return trainer.train()


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--labeled-json", default="")
    parser.add_argument("--labeled-root", default="")
    parser.add_argument("--unlabeled-json", default="")
    parser.add_argument("--unlabeled-root", default="")
    parser.add_argument("--val-json", default="")
    parser.add_argument("--val-root", default="")
    parser.add_argument("--class-names", nargs="+", default=["plant"])
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    print("Command Line Args:", parsed_args)
    launch(
        main,
        parsed_args.num_gpus,
        num_machines=parsed_args.num_machines,
        machine_rank=parsed_args.machine_rank,
        dist_url=parsed_args.dist_url,
        args=(parsed_args,),
    )
