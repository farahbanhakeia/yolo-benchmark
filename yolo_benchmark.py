"""
Module de benchmark pour comparer les différentes versions de YOLO.
"""

import time
import os
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import cv2

from model_factory import ModelFactory


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: list


@dataclass
class BenchmarkResult:
    model_name: str
    model_size_mb: float = 0.0
    load_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    num_detections: int = 0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 0.0
    detections: list = field(default_factory=list)
    fps: float = 0.0
    image_size: tuple = (0, 0)
    error: Optional[str] = None
    success: bool = True

    def to_dict(self):
        d = asdict(self)
        # Convertir le tuple en liste pour JSON
        d['image_size'] = list(d['image_size'])
        return d


# ─── Modeles disponibles ────────────────────────────────────
YOLO_MODELS = {
    "YOLOv4": {
    "weight": "yolov4.weights",
    "description": "YOLOv4"
},

"YOLOv4-Tiny": {
    "weight": "yolov4-tiny.weights",
    "description": "YOLOv4 Tiny"
},



    # YOLO26
    "YOLO26n": {
        "weight": "yolo26n.pt",
        "description": "YOLO26 Nano"
    },
    "YOLO26s": {
        "weight": "yolo26s.pt",
        "description": "YOLO26 Small"
    },
    "YOLO26m": {
        "weight": "yolo26m.pt",
        "description": "YOLO26 Medium"
    },
    

    # YOLOX
"YOLOX-N": {
    "weight": "yolox_m.pth",
    "description": "YOLOX N"
},
"YOLOX-S": {
    "weight": "yolox_s.pth",
    "description": "YOLOX Small"
},





    # YOLO-World
    "YOLO-World-S": {
        "weight": "yolov8s-world.pt",
        "description": "YOLO-World Small"
    },
    "YOLO-World-M": {
        "weight": "yolov8m-world.pt",
        "description": "YOLO-World Medium"
    },
    "YOLO-World-L": {
        "weight": "yolov8l-world.pt",
        "description": "YOLO-World Large"
    },
    "YOLO-World-XL": {
        "weight": "yolov8x-world.pt",
        "description": "YOLO-World XLarge"
    },

    # YOLOv12
    "YOLOv12n": {"weight": "yolo12n.pt", "description": "YOLOv12 Nano - Ultra rapide"},
    "YOLOv12s": {"weight": "yolo12s.pt", "description": "YOLOv12 Small - Bon compromis"},
    "YOLOv12m": {"weight": "yolo12m.pt", "description": "YOLOv12 Medium - Équilibre vitesse/précision"},
    "YOLOv12l": {"weight": "yolo12l.pt", "description": "YOLOv12 Large - Plus précis"},

    # YOLO11 - Nouveau nommage
    "YOLO11n": {"weight": "yolo11n.pt", "description": "YOLO11 Nano"},
    "YOLO11s": {"weight": "yolo11s.pt", "description": "YOLO11 Small"},
    "YOLO11m": {"weight": "yolo11m.pt", "description": "YOLO11 Medium"},
    "YOLO11l": {"weight": "yolo11l.pt", "description": "YOLO11 Large"},



    
    # YOLOv8
    "YOLOv8n": {"weight": "yolov8n.pt", "description": "YOLOv8 Nano - Ultra rapide"},
    "YOLOv8s": {"weight": "yolov8s.pt", "description": "YOLOv8 Small - Bon compromis"},
    "YOLOv8m": {"weight": "yolov8m.pt", "description": "YOLOv8 Medium - Equilibre"},
    "YOLOv8l": {"weight": "yolov8l.pt", "description": "YOLOv8 Large - Plus precis"},
    "YOLOv8x": {"weight": "yolov8x.pt", "description": "YOLOv8 XLarge - Max precision"},


    # YOLOv5
    "YOLOv5n": {"weight": "yolov5nu.pt", "description": "YOLOv5 Nano"},
    "YOLOv5s": {"weight": "yolov5su.pt", "description": "YOLOv5 Small"},
    "YOLOv5m": {"weight": "yolov5mu.pt", "description": "YOLOv5 Medium"},
    "YOLOv5l": {"weight": "yolov5lu.pt", "description": "YOLOv5 Large"},

    # YOLOv3
    "YOLOv3": {"weight": "yolov3u.pt", "description": "YOLOv3 Classique"},
    "YOLOv3-tiny": {"weight": "yolov3-tinyu.pt", "description": "YOLOv3 Tiny"},
}


class YOLOBenchmark:

    def __init__(self, upload_folder="uploads", results_folder="results"):
        self.upload_folder = Path(upload_folder)
        self.results_folder = Path(results_folder)
        self.upload_folder.mkdir(exist_ok=True)
        self.results_folder.mkdir(exist_ok=True)

    def get_available_models(self):
        models = []
        for name, info in YOLO_MODELS.items():
            models.append({
                "name": name,
                "weight": info["weight"],
                "description": info["description"]
            })
        return models

    def _get_model_size(self, model_name: str) -> float:
        size_map = {

            "YOLOv4":245,
            "YOLOv4-Tiny":23.1,

            "YOLOv12n": 5.33,
            "YOLOv12s": 30.0,
            "YOLOv12m": 18.1,
            "YOLOv12l": 51.2,

            "YOLO-World-S": 25.9,
            "YOLO-World-M": 55.8,
            "YOLO-World-L": 91.2,
            "YOLO-World-XL": 141,
            
            "YOLOX-N": 193.0,
            "YOLOX-S": 68.7,
            "YOLO26n": 5.28,
            "YOLO26s": 19.4,
            "YOLO26m": 42.2,
            "YOLOv8n": 6.2, "YOLOv8s": 22.5, "YOLOv8m": 52.0,
            "YOLOv8l": 87.7, "YOLOv8x": 136.7,
            
            "YOLOv5n": 3.9, "YOLOv5s": 14.1, "YOLOv5m": 40.8, "YOLOv5l": 89.3,
            "YOLO11n": 5.4, "YOLO11s": 18.4, "YOLO11m": 38.8, "YOLO11l": 49.0,
            "YOLOv3": 123.0, "YOLOv3-tiny": 17.0,
        }
        return size_map.get(model_name, 0.0)

    def benchmark_single(self, model_name: str, image_path: str,
                          conf_threshold: float = 0.25) -> BenchmarkResult:
        """Benchmark un seul modele sur une image."""

        result = BenchmarkResult(model_name=model_name)

        try:

            # ── Verifier le modele ──
            if model_name not in YOLO_MODELS:
                raise ValueError(f"Modele inconnu: {model_name}")

            # ── Verifier l'image ──
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image introuvable: {image_path}")

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de lire l'image: {image_path}")

            result.image_size = (int(img.shape[1]), int(img.shape[0]))
            result.model_size_mb = round(self._get_model_size(model_name), 2)

            print(f"   Image: {img.shape[1]}x{img.shape[0]}")

            weight = YOLO_MODELS[model_name]["weight"]
            start_load = time.perf_counter()
            adapter = ModelFactory.create(model_name, weight)
            adapter.load()
            load_time = (time.perf_counter() - start_load) * 1000
            result.load_time_ms = round(load_time, 2)

            print(f"   Modele charge en {result.load_time_ms}ms")

            # ── Warm-up ──
            print(f"   Warm-up...")
            try:
                adapter.predict(image_path, conf_threshold)
            except Exception as e:
                print(f"   Warm-up warning: {e}")

            # ── Inference (3 passes) ──
            print(f"   Inference...")
            inference_times = []
            last_results = None

            for run in range(3):
                start_inf = time.perf_counter()

                last_results = adapter.predict(
                    image_path,
                    conf_threshold
                )

                elapsed = (time.perf_counter() - start_inf) * 1000

                inference_times.append(elapsed)

                print(f"   Run {run+1}: {elapsed:.1f}ms")

            avg_inference = float(np.mean(inference_times))
            result.inference_time_ms = round(avg_inference, 2)
            result.total_time_ms = round(result.load_time_ms + avg_inference, 2)
            result.fps = round(1000.0 / avg_inference, 2) if avg_inference > 0 else 0

            if last_results is not None:

                boxes = last_results["boxes"]
                confidences = last_results["scores"]
                class_ids = last_results["classes"]
                names = last_results["names"]

                detections = []

                for i in range(len(boxes)):

                    class_name = names.get(
                        int(class_ids[i]),
                        str(class_ids[i])
                    )

                    bbox = [float(x) for x in boxes[i]]

                    detections.append({
                        "class_name": class_name,
                        "confidence": round(float(confidences[i]), 4),
                        "bbox": [round(x, 1) for x in bbox]
                    })

                result.detections = detections
                result.num_detections = len(detections)

                if len(confidences):
                    result.avg_confidence = round(float(np.mean(confidences)), 4)
                    result.max_confidence = round(float(np.max(confidences)), 4)
                    result.min_confidence = round(float(np.min(confidences)), 4)

                print(f"   Detections : {result.num_detections}")

            else:
                print("   Aucune detection")

            # ── Liberer la memoire ──
            adapter.release()

            result.success = True
            print(f"   ✅ Termine!")

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)}"
            print(f"   ❌ ERREUR: {result.error}")
            traceback.print_exc()

        return result

    def benchmark_multiple(self, model_names: list, image_path: str,
                            conf_threshold: float = 0.25) -> list:
        results = []
        total = len(model_names)

        for i, name in enumerate(model_names):
            print(f"\n{'='*50}")
            print(f"[{i+1}/{total}] Benchmarking {name}")
            print(f"{'='*50}")

            result = self.benchmark_single(name, image_path, conf_threshold)
            results.append(result.to_dict())

        return results

    def save_results(self, results: list, filename: str = "benchmark_results.json"):
        filepath = self.results_folder / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return str(filepath)

    def generate_annotated_image(self, image_path: str, detections: list,
                                  model_name: str) -> str:
        """Genere une image annotee avec les detections."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"   Cannot read image for annotation")
                return ""

            np.random.seed(42)
            colors = {}

            for det in detections:
                cls = det['class_name']
                if cls not in colors:
                    colors[cls] = tuple([int(c) for c in np.random.randint(50, 255, 3)])

                bbox = det['bbox']
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                color = colors[cls]

                # Rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{cls} {det['confidence']:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Fond du label
                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

            # Sauvegarder
            safe_name = model_name.replace(" ", "_").replace("/", "_")
            stem = Path(image_path).stem
            output_name = f"annotated_{safe_name}_{stem}.jpg"
            output_path = str(self.results_folder / output_name)

            success = cv2.imwrite(output_path, img)
            if success:
                print(f"   Image annotee: {output_name}")
                return output_name
            else:
                print(f"   Echec ecriture image")
                return ""

        except Exception as e:
            print(f"   Erreur annotation: {e}")
            traceback.print_exc()
            return ""