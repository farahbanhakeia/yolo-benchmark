import numpy as np

from ultralytics import YOLO

from .base_adapter import BaseAdapter


class UltralyticsAdapter(BaseAdapter):

    def load(self):
        self.model = YOLO(self.weight_path)

    def predict(self, image_path, conf=0.25):

        results = self.model.predict(
            source=image_path,
            conf=conf,
            verbose=False
        )

        if len(results) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "names": {}
            }

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:

            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "classes": np.array([]),
                "names": r.names
            }

        return {

            "boxes": r.boxes.xyxy.cpu().numpy(),

            "scores": r.boxes.conf.cpu().numpy(),

            "classes": r.boxes.cls.cpu().numpy().astype(int),

            "names": r.names
        }

    def release(self):

        del self.model

        try:

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass