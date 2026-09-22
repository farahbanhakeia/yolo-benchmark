import os
import sys
import cv2
import torch
import numpy as np

from .base_adapter import BaseAdapter

YOLOX_ROOT = os.path.abspath("external/YOLOX")

if YOLOX_ROOT not in sys.path:
    sys.path.insert(0, YOLOX_ROOT)

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


class YOLOXAdapter(BaseAdapter):

    def __init__(self, weight_path):
        super().__init__(weight_path)

        self.model = None
        self.exp = None
        self.preproc = ValTransform(legacy=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_names = {
            i: name for i, name in enumerate([
                "person","bicycle","car","motorcycle","airplane",
                "bus","train","truck","boat","traffic light",
                "fire hydrant","stop sign","parking meter","bench","bird",
                "cat","dog","horse","sheep","cow",
                "elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee",
                "skis","snowboard","sports ball","kite","baseball bat",
                "baseball glove","skateboard","surfboard","tennis racket",
                "bottle","wine glass","cup","fork","knife",
                "spoon","bowl","banana","apple","sandwich",
                "orange","broccoli","carrot","hot dog","pizza",
                "donut","cake","chair","couch","potted plant",
                "bed","dining table","toilet","tv","laptop",
                "mouse","remote","keyboard","cell phone","microwave",
                "oven","toaster","sink","refrigerator","book",
                "clock","vase","scissors","teddy bear","hair drier",
                "toothbrush"
            ])
        }

    # -------------------------
    # LOAD MODEL
    # -------------------------
    def load(self):

        filename = os.path.basename(self.weight_path).lower()
        print("YOLOX weight:", filename)

        if "yolox_n" in filename or "yolox-n" in filename:
            exp_file = os.path.join(YOLOX_ROOT, "exps", "default", "yolox_n.py")
        elif "yolox_s" in filename or "yolox-s" in filename:
            exp_file = os.path.join(YOLOX_ROOT, "exps", "default", "yolox_s.py")
        elif "yolox_m" in filename or "yolox-m" in filename:
            exp_file = os.path.join(YOLOX_ROOT, "exps", "default", "yolox_m.py")
        elif "yolox_l" in filename or "yolox-l" in filename:
            exp_file = os.path.join(YOLOX_ROOT, "exps", "default", "yolox_l.py")
        else:
            exp_file = os.path.join(YOLOX_ROOT, "exps", "default", "yolox_x.py")

        self.exp = get_exp(exp_file, None)
        self.model = self.exp.get_model()

        ckpt = torch.load(self.weight_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # PREDICT (FIXED + SAFE)
    # -------------------------
    def predict(self, image_path, conf=0.25):

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image introuvable: {image_path}")

        # -------------------------
        # PREPROCESS SAFE
        # -------------------------
        img_info = self.preproc(image, None, self.exp.test_size)

        if isinstance(img_info, tuple):
            img = img_info[0]
            ratio = img_info[1] if len(img_info) > 1 else 1.0
        else:
            img = img_info
            ratio = 1.0

        # sécurisation ratio
        try:
            ratio = float(np.array(ratio).reshape(-1)[0])
        except Exception:
            ratio = 1.0

        if ratio == 0 or np.isnan(ratio) or np.isinf(ratio):
            ratio = 1.0

        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        # -------------------------
        # INFERENCE
        # -------------------------
        with torch.no_grad():
            outputs = self.model(img)

            outputs = postprocess(
                outputs,
                self.exp.num_classes,
                conf,
                self.exp.nmsthre
            )

        # -------------------------
        # NO DETECTIONS
        # -------------------------
        if outputs[0] is None:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty((0,)),
                "classes": np.empty((0,), dtype=int),
                "names": self.class_names
            }

        # -------------------------
        # POSTPROCESS SAFE
        # -------------------------
        detections = outputs[0].cpu().numpy()

        # 🔥 IMPORTANT FIX GLOBAL CLEANUP (avant toute division)
        detections = np.nan_to_num(detections, nan=0.0, posinf=0.0, neginf=0.0)

        boxes = detections[:, :4]

        # éviter division dangereuse (double sécurité, déjà clampé plus haut)
        if ratio == 0 or np.isnan(ratio) or np.isinf(ratio):
            ratio = 1.0

        boxes = boxes / ratio
        boxes = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)

        scores = detections[:, 4]
        classes = detections[:, 5].astype(int)

        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "names": self.class_names
        }

    # -------------------------
    # CLEANUP
    # -------------------------
    def release(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()