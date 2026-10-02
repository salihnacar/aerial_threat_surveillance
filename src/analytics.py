import numpy as np
import cv2

class ThreatHeatmap:
    def __init__(self, width, height, decay_rate=0.99):
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.accumulation_matrix = np.zeros((self.height, self.width), dtype=np.float32)

    def update_threats(self, predictions):
        self.accumulation_matrix *= self.decay_rate
        for pred in predictions:
            bbox = pred.bbox
            center_x = int((bbox.minx + bbox.maxx) / 2)
            center_y = int((bbox.miny + bbox.maxy) / 2)
            if 0 <= center_x < self.width and 0 <= center_y < self.height:
                self.accumulation_matrix[center_y, center_x] += 50.0

    def apply_heatmap_overlay(self, frame, alpha=0.6):
        heat_clipped = np.clip(self.accumulation_matrix, 0, 255).astype(np.uint8)
        heat_blurred = cv2.GaussianBlur(heat_clipped, (75, 75), 0)
        color_heatmap = cv2.applyColorMap(heat_blurred, cv2.COLORMAP_JET)
        
        # Mask logic for clean overlay
        mask = (heat_blurred > 10)[..., np.newaxis]
        overlay = np.where(mask, (frame * (1 - alpha) + color_heatmap * alpha).astype(np.uint8), frame)
        
        return overlay