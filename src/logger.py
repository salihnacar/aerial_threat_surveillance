import sqlite3
import os
import cv2
from datetime import datetime

class ThreatDatabaseLogger:
    def __init__(self, db_path, crop_dir):
        self.db_path = db_path
        self.crop_dir = crop_dir
        os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS threats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, label TEXT, confidence REAL,
                bbox TEXT, image_path TEXT
            )
        ''')
        self.conn.commit()
        self.last_log_time = {}

    def log_threat(self, frame, bbox_coords, label, confidence, cooldown_seconds=2.0):
        current_time = datetime.now()
        if label in self.last_log_time:
            if (current_time - self.last_log_time[label]).total_seconds() < cooldown_seconds:
                return

        x1, y1, x2, y2 = bbox_coords
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")
        crop_path = os.path.join(self.crop_dir, f"{label}_{timestamp_str}.jpg")
        
        threat_crop = frame[y1:y2, x1:x2]
        if threat_crop.size > 0:
            cv2.imwrite(crop_path, threat_crop)

        bbox_str = f"[{x1}, {y1}, {x2}, {y2}]"
        self.cursor.execute('''
            INSERT INTO threats (timestamp, label, confidence, bbox, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (current_time.strftime("%Y-%m-%d %H:%M:%S"), label, confidence, bbox_str, crop_path))
        self.conn.commit()
        self.last_log_time[label] = current_time
        print(f"[DATABASE] Logged {label} ({confidence:.2f}) to {crop_path}")

    def close(self):
        self.conn.close()