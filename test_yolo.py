"""
Script de diagnostic - Testez ceci AVANT de lancer l'app Flask
"""

import sys
print(f"Python: {sys.version}")
print()

# 1. Test des imports
print("=" * 50)
print("TEST DES IMPORTS")
print("=" * 50)

try:
    import numpy as np
    print(f"  numpy:    OK ({np.__version__})")
except ImportError as e:
    print(f"  numpy:    ERREUR - {e}")
    print("  → pip install numpy")

try:
    import cv2
    print(f"  opencv:   OK ({cv2.__version__})")
except ImportError as e:
    print(f"  opencv:   ERREUR - {e}")
    print("  → pip install opencv-python-headless")

try:
    import torch
    print(f"  torch:    OK ({torch.__version__})")
    print(f"  CUDA:     {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  torch:    ERREUR - {e}")
    print("  → pip install torch torchvision")

try:
    import ultralytics
    print(f"  ultralytics: OK ({ultralytics.__version__})")
except ImportError as e:
    print(f"  ultralytics: ERREUR - {e}")
    print("  → pip install ultralytics")

try:
    import flask
    print(f"  flask:    OK ({flask.__version__})")
except ImportError as e:
    print(f"  flask:    ERREUR - {e}")
    print("  → pip install flask")

print()

# 2. Test YOLO rapide
print("=" * 50)
print("TEST YOLO RAPIDE")
print("=" * 50)

try:
    from ultralytics import YOLO
    import time
    import os
    import urllib.request

    # Telecharger une image de test
    test_image = "test_bus.jpg"
    if not os.path.exists(test_image):
        print("  Telechargement image de test...")
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, test_image)
        print(f"  Image telechargee: {test_image}")

    # Verifier l'image
    img = cv2.imread(test_image)
    if img is None:
        print(f"  ERREUR: Impossible de lire {test_image}")
        sys.exit(1)
    print(f"  Image: {img.shape[1]}x{img.shape[0]}")

    # Charger YOLOv8n (le plus petit)
    print("  Chargement YOLOv8n...")
    start = time.time()
    model = YOLO("yolov8n.pt")
    load_time = time.time() - start
    print(f"  Charge en {load_time:.1f}s")

    # Inference
    print("  Inference...")
    start = time.time()
    results = model.predict(source=test_image, conf=0.25, verbose=False)
    inf_time = time.time() - start
    print(f"  Inference en {inf_time*1000:.1f}ms")

    # Resultats
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            num = len(r.boxes)
            print(f"  Detections: {num}")

            if num > 0:
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(min(5, num)):
                    name = r.names.get(int(classes[i]), "?")
                    print(f"    - {name}: {confs[i]:.2f}")

            print()
            print("  ✅ TOUT FONCTIONNE!")
            print("  Vous pouvez lancer: python app.py")
        else:
            print("  Aucune detection (normal si seuil trop haut)")
            print("  ✅ YOLO fonctionne quand meme!")
    else:
        print("  Pas de resultats")

except Exception as e:
    print(f"\n  ❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("  SOLUTIONS:")
    print("  1. pip install --upgrade ultralytics")
    print("  2. pip install --upgrade torch torchvision")
    print("  3. pip install opencv-python-headless")