import tkinter as tk
from tkinter import ttk
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
from prediction_tracking_module import Predictor


class PedestrianTrajectoryPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian trajectory prediction")

        # Variables for video and plotting
        self.video_source = "data/videos/20231109-112534_10s.mp4"
        self.cap = cv2.VideoCapture(self.video_source)
        self.paused = False
        HEIGHT, WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        # Create video canvas
        self.video_canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT)
        self.video_canvas.grid(row=0, column=0, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(4, 8), dpi=100)
        self.ax.imshow(
            cv2.cvtColor(cv2.imread("data/bg.png"), cv2.COLOR_BGR2RGB),
            extent=[0, 8.5, 0, 32],
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_xticks(np.arange(0, 8.1, 1))
        self.ax.set_yticks(
            np.arange(0, 32.1, 1), labels=[None if i % 2 else i for i in range(33)]
        )
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(32, 0)
        self.ax.grid()

        # Create canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

        # Create buttons and radio buttons
        self.pause_button = tk.Button(
            root, text="Pause video", command=self.toggle_pause
        )
        self.pause_button.grid(row=1, column=0, pady=10)

        options = ["Predict 1s", "Predict 3s", "Predict 5s"]
        self.selected_option = tk.StringVar(value=options[0])
        for i, option in enumerate(options):
            radio_button = ttk.Radiobutton(
                root, text=option, variable=self.selected_option, value=option
            )
            radio_button.grid(row=i + 2, column=0, sticky=tk.W, pady=5, padx=10)

        self.predicting_model = "model_td_20lstm.h5"
        self.predictor = Predictor(self.predicting_model, self.cap)
        self.prediction = self.predictor.predict()
        self.colors = []

        self.update()

    def plot_people_on_plan(self, people, i, predictions=None):
        def color(track_id):
            if track_id not in self.colors:
                self.colors.append(track_id)
            return self.colors.index(track_id)

        if people is not None:
            for track_id, (x, y) in people.items():
                plt.scatter(
                    x,
                    y,
                    label=f"Person {track_id}",
                    marker="o",
                    color=f"C{color(track_id)}",
                )
                if predictions is not None and track_id in predictions:
                    plt.plot(
                        predictions[track_id][:10, 0],
                        predictions[track_id][:10, 1],
                        label=f"Person {track_id} prediction",
                        color=f"C{color(track_id)}",
                    )

    def update(self):
        if not self.paused:
            try:
                frame, *people = next(self.prediction)

                self.photo = self.convert_to_tk_image(frame)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                self.plot_people_on_plan(*people)
                self.canvas.draw_idle()
            except StopIteration:
                self.predictor.tracker.stop = True
                self.cap.release()
                self.root.after_cancel(self.update)
                tk.messagebox.showinfo("Video ended", "Video ended")
                return

        self.root.after(5, self.update)

    def convert_to_tk_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img)

    def toggle_pause(self):
        self.paused = not self.paused

    def __del__(self):
        if hasattr(self, "cap"):
            self.predictor.tracker.stop = True
            self.cap.release()


def on_closing():
    response = tk.messagebox.askyesno("Exit", "Are you sure you want to exit?")
    if response:
        root.after_cancel(app.update)
        root.withdraw()
        root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianTrajectoryPredictor(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()
