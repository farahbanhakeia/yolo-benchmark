import cv2
import numpy as np

from .base_adapter import BaseAdapter


class YOLOv4Adapter(BaseAdapter):

    def __init__(self, weight_path):
        super().__init__(weight_path)

        self.net = None
        self.output_layers = None
        self.class_names = {}

    def load(self):

        if "tiny" in self.weight_path.lower():
            cfg = "yolov4-tiny.cfg"
        else:
            cfg = "yolov4.cfg"

        self.net = cv2.dnn.readNet(
            self.weight_path,
            cfg
        )

        self.net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_OPENCV
        )

        self.net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CPU
        )

        layer_names = self.net.getLayerNames()

        self.output_layers = [
            layer_names[i - 1]
            for i in self.net.getUnconnectedOutLayers().flatten()
        ]

        coco = [
            "person","bicycle","car","motorbike","aeroplane",
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
            "donut","cake","chair","sofa","pottedplant",
            "bed","diningtable","toilet","tvmonitor","laptop",
            "mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book",
            "clock","vase","scissors","teddy bear","hair drier",
            "toothbrush"
        ]

        self.class_names = {
            i: coco[i]
            for i in range(len(coco))
        }

    def predict(self, image_path, conf=0.25):

        image = cv2.imread(image_path)

        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)

        outputs = self.net.forward(
            self.output_layers
        )

        boxes = []
        scores = []
        classes = []

        for output in outputs:

            for detection in output:

                scores_all = detection[5:]

                class_id = np.argmax(scores_all)

                confidence = scores_all[class_id]

                if confidence < conf:
                    continue

                cx = detection[0] * w
                cy = detection[1] * h
                bw = detection[2] * w
                bh = detection[3] * h

                x = cx - bw / 2
                y = cy - bh / 2

                boxes.append([
                    x,
                    y,
                    x + bw,
                    y + bh
                ])

                scores.append(float(confidence))

                classes.append(int(class_id))

        if len(boxes):

            indices = cv2.dnn.NMSBoxes(
                [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
                scores,
                conf,
                0.45
            )

            final_boxes = []
            final_scores = []
            final_classes = []

            if len(indices):

                for idx in indices.flatten():

                    final_boxes.append(boxes[idx])

                    final_scores.append(scores[idx])

                    final_classes.append(classes[idx])

            boxes = np.array(final_boxes)

            scores = np.array(final_scores)

            classes = np.array(final_classes)

        else:

            boxes = np.empty((0,4))

            scores = np.empty((0,))

            classes = np.empty((0,),dtype=int)

        return {

            "boxes": boxes,

            "scores": scores,

            "classes": classes,

            "names": self.class_names

        }

    def release(self):

        self.net = None