from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
from ..components.utils import DistanceConverter


class Tracker:
    def __init__(self, cap, draw_tracking_lines=True):
        self.model = YOLO("yolov8n.pt")
        self.cap = cap
        self.track_history = defaultdict(lambda: [])
        self.track_history_m = defaultdict(lambda: [])
        self.draw_tracking_lines = draw_tracking_lines
        self.area = np.array([[[0, 151], [373, 151], [968, 555], [0, 555]]])
        self.people_at_frame = defaultdict(lambda: {})
        self.frame_number = 0

    def track(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            self.frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            if success:
                results = self.model.track(
                    frame, persist=True, classes=[0], verbose=False
                )

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                annotated_frame = results[0].plot(conf=False)

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    x, y = float(x), float(y)
                    track = self.track_history[track_id]
                    track.append((x, y))
                    if len(track) > 90:
                        track.pop(0)

                    if cv2.pointPolygonTest(self.area, (x, y), False) > 0:
                        x_meters, y_meters = DistanceConverter.pixels2meters(x, y)

                        self.track_history_m[track_id].append((x_meters, y_meters))
                        self.people_at_frame[self.frame_number][track_id] = (
                            x_meters,
                            y_meters,
                        )

                        cv2.putText(
                            annotated_frame,
                            f"x={x_meters:.2f}m, y={y_meters:.2f}m",
                            (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                        if self.draw_tracking_lines:
                            points = (
                                np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            )
                            cv2.polylines(
                                annotated_frame,
                                [points],
                                isClosed=False,
                                color=(230, 230, 230),
                                thickness=10,
                            )

                        alpha = 0.1
                        overlay = annotated_frame.copy()

                        cv2.polylines(
                            overlay,
                            pts=self.area,
                            isClosed=True,
                            color=(255, 0, 0),
                            thickness=2,
                        )
                        cv2.fillPoly(overlay, self.area, (255, 0, 0))
                        annotated_frame = cv2.addWeighted(
                            overlay, alpha, annotated_frame, 1 - alpha, 0
                        )

                yield (
                    annotated_frame,
                    self.people_at_frame[self.frame_number],
                    self.track_history_m,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            else:
                break

        self.cap.release()
