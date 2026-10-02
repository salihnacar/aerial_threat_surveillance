import argparse
import cv2
import torch
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.analytics import ThreatHeatmap
from src.logger import ThreatDatabaseLogger

def get_args():
    parser = argparse.ArgumentParser(description="Aerial Drone Surveillance Pipeline")
    parser.add_argument("--input", type=str, default=os.path.join("data", "input", "input.mp4"))
    parser.add_argument("--output", type=str, default=os.path.join("data", "output", "output.mp4"))
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    print(f"[SYSTEM] Initializing on {device.upper()}")

    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=args.weights,
        confidence_threshold=0.3, device=device
    )

    cap = cv2.VideoCapture(args.input)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize Pro Modules
    heatmap_tracker = ThreatHeatmap(width, height)
    db_logger = ThreatDatabaseLogger(os.path.join("data", "threat_logs.db"), os.path.join("data", "crops"))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. Inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            rgb_frame, model, slice_height=512, slice_width=512,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
        )

        # 2. Update Heatmap
        heatmap_tracker.update_threats(result.object_prediction_list)

        # 3. Draw & Log
        for pred in result.object_prediction_list:
            b = pred.bbox
            x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, pred.category.name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            db_logger.log_threat(frame, (x1, y1, x2, y2), pred.category.name, pred.score.value)

        # 4. Apply Heatmap
        frame = heatmap_tracker.apply_heatmap_overlay(frame)
        out.write(frame)
        
        if frame_count % 30 == 0: print(f"[PROFILER] Processed {frame_count}/{total_frames}")

    cap.release()
    out.release()
    db_logger.close()
    print(f"[SUCCESS] Output saved to {args.output}")

if __name__ == "__main__":
    main()