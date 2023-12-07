import cv2
import numpy as np
from collections import defaultdict
import tensorflow as tf


class Predictor:
    def __init__(self, model):
        self.model = tf.keras.models.load_model(model)
        self.track_history = defaultdict(lambda: [])
        self.num_steps_to_predict = 19

    def update(self, track_history):
        self.track_history = track_history

    def predict(self):
        return {
            person: self.model.predict(
                np.array(track[-self.num_steps_to_predict :]).reshape(
                    1, self.num_steps_to_predict, 2
                ),
                verbose=0,
            )[0]
            for person, track in self.track_history.items()
            if len(track) > self.num_steps_to_predict
        }
