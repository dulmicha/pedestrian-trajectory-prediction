import cv2
import numpy as np
from collections import defaultdict
import tensorflow as tf
from ultralytics import YOLO


class Predictor:
    def __init__(self, model, cap):
        self.model = tf.keras.models.load_model(model)
        self.cap = cap
        self.track_history = defaultdict(lambda: [])
        self.area = np.array([[[0, 151], [383, 151], [978, 555], [0, 555]]])
        self.tracker = Tracker(cap)
        self.people_at_frame = defaultdict(lambda: {})

    def predict(self):
        for frame_number, (frame, boxes, track_ids) in enumerate(self.tracker.track()):
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x, y = int(x), int(y)

                if cv2.pointPolygonTest(self.area, (x, y), False) > 0:
                    self.track_history[track_id].append(
                        DistanceConverter.pixels2meters(x, y)
                    )
                    self.people_at_frame[frame_number][
                        track_id
                    ] = DistanceConverter.pixels2meters(x, y)

            if frame_number % 10 == 0:
                predictions = {
                    person: self.model.predict(
                        np.array(track[-19:]).reshape(1, 19, 2), verbose=0
                    )[0]
                    for person, track in self.track_history.items()
                    if len(track) > 19
                }
                yield (
                    frame,
                    self.people_at_frame[frame_number],
                    frame_number,
                    predictions,
                )

            if self.tracker.stop:
                break


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

        # self.cap.release()


class DistanceConverter:
    @staticmethod
    def pixels2meters(x, y):
        X_px2meters = (
            lambda x, y: (1 - ((1.47 * y + 161.76 - x) / (1.47 * y + 161.76))) * 8
        )
        Y_px2meters = (
            lambda y: 6.16793058e-07 * y**3
            - 8.61522438e-04 * y**2
            + 4.31688489e-01 * y
            - 4.75010213e01
        )
        return X_px2meters(x, y), Y_px2meters(y)

    @staticmethod
    def meters2pixels(x, y):
        X_meters2px = lambda x, y: (1.47 * y + 161.76) * x / 8
        Y_meters2px = (
            lambda y: 1.51690315e-02 * y**3
            - 3.20503299e-01 * y**2
            + 7.27107405e00 * y
            + 1.47945378e02
        )
        y_px = Y_meters2px(y)
        return X_meters2px(x, y_px), y_px
