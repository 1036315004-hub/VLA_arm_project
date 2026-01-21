# src/perception/yolo_detector.py
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", imgsz: int = 320, conf: float = 0.25):
        """
        model_path: 权重（yolov8n.pt）
        device: "cpu" 或 "cuda:0"
        imgsz: 推理输入分辨率（建议 320 或 416）
        conf: 最低置信度阈值
        """
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        # ultralytics 的 predict 可以直接接 numpy
        # 但为了稳定，先 resize 保证输入分辨率
    def predict(self, img: np.ndarray) -> List[Dict]:
        """
        img: HxWx3 RGB uint8 numpy
        返回 list of {'bbox': (x1,y1,x2,y2), 'score': float, 'label': str}
        bbox 为原图坐标
        """
        # ultralytics handles resizing internally and returns coordinates mapped to original image
        # Convert numpy to PIL to avoid torch.from_numpy issues in broken envs
        if isinstance(img, np.ndarray):
            source = Image.fromarray(img)
        else:
            source = img

        results = self.model.predict(source=source, device=self.device, imgsz=self.imgsz, conf=self.conf, verbose=False)
        detections = []
        # results 是一个 list（batch），我们只传一张图
        r = results[0]
        # boxes xyxy in original image
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().tolist()  # x1,y1,x2,y2
                score = float(box.conf[0].item())
                cls = int(box.cls[0].item())
                # map class id to name if model.names exists
                label = self.model.names.get(cls, str(cls)) if hasattr(self.model, 'names') else str(cls)

                x1, y1, x2, y2 = xyxy
                detections.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "score": score, "label": label})
        return detections