
# -*- coding: UTF-8 -*-
import os
import sys
import warnings
# warnings.filterwarnings("ignore")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse
from flask import Flask, jsonify, request, url_for
from flask_cors import CORS

from loguru import logger
from detectron2.data.detection_utils import read_image
from predictor import VisualizationDemo
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
    confidence_threshold: float = 0.275
    opts: list = None
    
args = ModelConfig(
    config_file="configs/xunfei/R_50_poly_ft.yaml",
    opts=["MODEL.WEIGHTS", "output/para_detr_r50_ft/model_0051999.pth"]
)
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)

###############################################################
###############################################################

###########
## flask ##
###########

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = 'secret to test!'
GUID = 'de683f7810e84046ab5a8240a7cc0be3'  # 用于测试的guid


@app.route("/api/detect_cell", methods=["POST"])
def detect():

    try:
        img = request.form["image_path"]
    except:
        return jsonify({"code": 100})
    # # base64解析
    # img = base64.b64decode(str(img))
    # image_data = np.frombuffer(img, np.uint8)
    # image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # print(image_data)
    # predict
    # use PIL, to be consistent with evaluation
    img = read_image(img, format="BGR")

    # polygons n*32 左上角开始的16个点顺时针
    predictions, _ = demo.run_on_image(img, vis=False)
    polygons = predictions['instances'].polygons.cpu().numpy()

    # 返回结果
    res_info = [] # n*m*2 m=16
    for polygon in polygons:
        polygon = polygon.reshape(-1, 2).tolist()
        polygon = [[float(p[0]), float(p[1])] for p in polygon]
        res_info.append(polygon)
    logger.info(f"detect {len(res_info)} cells")
    final_result = {
        "result": res_info,
        "code": 200,
    }
    final_result = jsonify(final_result)
    return final_result


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19500)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
