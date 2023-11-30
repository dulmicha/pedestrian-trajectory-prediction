import cv2
import numpy as np
from collections import defaultdict
import tensorflow as tf
from ..components.track import Tracker
from ..components.utils import DistanceConverter


class Predictor:
    def __init__(self, model, cap):
        self.model = tf.keras.models.load_model(model)
        self.cap = cap
        self.track_history = defaultdict(lambda: [])
        self.area = np.array([[[0, 151], [383, 151], [978, 555], [0, 555]]])
        self.tracker = Tracker(cap, draw_tracking_lines=False)
        self.people_at_frame = defaultdict(lambda: {})
        self.num_steps_to_predict = 19

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
                        np.array(track[-self.num_steps_to_predict:]).reshape(1, self.num_steps_to_predict, 2), verbose=0
                    )[0]
                    for person, track in self.track_history.items()
                    if len(track) > self.num_steps_to_predict
                }
                yield (
                    frame,
                    self.people_at_frame[frame_number],
                    frame_number,
                    predictions,
                )

            if self.tracker.stop:
                break
