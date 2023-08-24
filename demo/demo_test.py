# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image

from predictor import VisualizationDemo
from detectron2.modeling import GeneralizedRCNNWithTTA
from adet.config import get_cfg
from dataclasses import dataclass


###############################################################
###############################################################


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

@dataclass
class ModelConfig:
    config_file: str = None
    confidence_threshold: float = 0.3
    opts: list = None
    
args = ModelConfig(
    config_file="configs/xunfei/R_50_poly.yaml",
    opts=["MODEL.WEIGHTS", "output/para_detr_r50/model_final.pth"]
)
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)
demo.predictor.model = GeneralizedRCNNWithTTA(demo.predictor.cfg, demo.predictor.model)
demo.predictor.model.eval()

###############################################################
###############################################################

def predict(img_path):
    # use PIL, to be consistent with evaluation
    img = read_image(img_path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img, vis=False)
    polygons = predictions['instances'].polygons.cpu().numpy()
    polygon = polygons[0]
    polygon = polygon.reshape(-1, 2).tolist()
    print(polygon)
    # visualized_output.save("./output/visual.png")


if __name__ == "__main__":
    predict("../ppocr_data/images/00001.png")