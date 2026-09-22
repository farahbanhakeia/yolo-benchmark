"""
Factory permettant de créer automatiquement le bon adaptateur
pour chaque famille de modèles YOLO.
"""

from adapters.ultralytics_adapter import UltralyticsAdapter
from adapters.yolov4_adapter import YOLOv4Adapter
from adapters.yolox_adapter import YOLOXAdapter
class ModelFactory:

    @staticmethod
    def create(model_name: str, weight_path: str):

        name = model_name.lower()

        # -----------------------------
        # YOLOv4
        # -----------------------------
        if "yolov4" in name:
            return YOLOv4Adapter(weight_path)
         # -----------------------------
        # YOLOX
        # -----------------------------
        elif "yolox" in name:
            return YOLOXAdapter(weight_path)

        

        # -----------------------------
        # Tous les modèles Ultralytics
        # -----------------------------
        elif (
            "yolov3" in name or
            "yolov5" in name or
            "yolov6" in name or
            "yolov7" in name or
            "yolov8" in name or
            "yolo11" in name or
            "yolo12" in name or
            "yolov12" in name or

            "yolo26" in name or
            "yolov26" in name or

            "world" in name
        ):
            return UltralyticsAdapter(weight_path)

        # -----------------------------
        # Modèle inconnu
        # -----------------------------
        raise ValueError(
            f"Aucun adaptateur disponible pour : {model_name}"
        )