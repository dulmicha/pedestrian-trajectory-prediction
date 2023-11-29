from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
from ..components.utils import DistanceConverter


class Tracker:
    def __init__(self, cap):
        self.model = YOLO("yolov8n.pt")
        self.cap = cap
        self.track_history = defaultdict(lambda: [])
        self.frames_list = defaultdict(lambda: [])
        self.stop = False

    def track(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                results = self.model.track(
                    frame, persist=True, classes=[0], verbose=False
                )

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                annotated_frame = results[0].plot(conf=False)

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 90:
                        track.pop(0)

                    frames = self.frames_list[track_id]
                    frames.append(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if len(frames) > 3:
                        frames.pop(0)

                    x_meters, y_meters = DistanceConverter.pixels2meters(x, y)
                    cv2.putText(
                        annotated_frame,
                        f"x={x_meters:.2f}m, y={y_meters:.2f}m",
                        (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )

                    area = np.array([[[0, 151], [373, 151], [968, 555], [0, 555]]])
                    alpha = 0.1
                    overlay = annotated_frame.copy()

                    cv2.polylines(
                        overlay, pts=area, isClosed=True, color=(255, 0, 0), thickness=2
                    )
                    cv2.fillPoly(overlay, area, (255, 0, 0))
                    annotated_frame = cv2.addWeighted(
                        overlay, alpha, annotated_frame, 1 - alpha, 0
                    )

                yield (annotated_frame, boxes, track_ids)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if self.stop:
                    break
            else:
                break

        self.cap.release()
