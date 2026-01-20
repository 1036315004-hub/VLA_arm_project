# src/perception/vlm_yolo_clip.py
from .yolo_detector import YOLODetector
from .clip_ranker import CLIPRanker
import numpy as np
import cv2
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLM_YOLO_CLIP")

class VLM_YOLO_CLIP:
    def __init__(self, yolo_model="yolov8n.pt", device="cpu", imgsz=320, conf=0.25, clip_model="ViT-B-32"):
        self.detector = YOLODetector(model_path=yolo_model, device=device, imgsz=imgsz, conf=conf)
        self.ranker = CLIPRanker(model_name=clip_model, device=device)

    def query_image(self, rgb: np.ndarray, text_query: str, topk: int = 1) -> List[Dict]:
        """
        rgb: HxWx3 RGB uint8 numpy
        text_query: e.g. "red cup"
        返回 topk 个检测，格式 [{'bbox':..., 'score':..., 'label':..., 'clip_score':...}]
        流程: YOLO 提取候选 -> 对每个候选 crop 用 CLIP score 排序 -> 返回 topk
        """
        logger.info(f"Querying image with text: '{text_query}'")
        dets = self.detector.predict(rgb)
        if not dets:
            logger.info("No YOLO detections found.")
            return []

        # build crops
        crops = []
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            # pad & crop safely
            h,w = rgb.shape[:2]
            x1s = max(0, x1); y1s = max(0, y1)
            x2s = min(w-1, x2); y2s = min(h-1, y2)
            crop = rgb[y1s:y2s, x1s:x2s]
            if crop.size == 0:
                crops.append(cv2.resize(rgb, (224,224)))
            else:
                crops.append(cv2.resize(crop, (224,224)))

        # clip rank
        clip_scores = self.ranker.score_crops(crops, text_query)
        # combine
        for i,d in enumerate(dets):
            d["clip_score"] = clip_scores[i] if i < len(clip_scores) else 0.0
        # sort by clip_score desc * confidence
        dets_sorted = sorted(dets, key=lambda x: (x.get("clip_score",0)*0.7 + x.get("score",0)*0.3), reverse=True)
        return dets_sorted[:topk]