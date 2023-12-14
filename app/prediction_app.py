import tkinter as tk
import customtkinter as ctk
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
from .components.predict import Predictor
from .components.track import Tracker
import time
import os

DEFAULT_MODEL_PATH = os.path.join("models", "model_td_100lstm.h5")
DEFAULT_VIDEO_PATH = os.path.join("data", "default_video.mp4")
BACKGROUND_IMAGE_PATH = os.path.join("data", "bg.png")


class PedestrianTrajectoryPredictionApp:
    def __init__(self, root, predicting_model_path=None):
        self.root = root
        self.root.title("Pedestrian trajectory prediction")
        self.root.resizable(False, False)

        # Variables for video and plotting
        response = tk.messagebox.askyesno(
            "Video selection", "Do you want to use default video?"
        )
        if not response:
            video_path = tk.filedialog.askopenfilename(
                initialdir="data",
                title="Select file",
                filetypes=[("mp4 files", "*.mp4")],
            )
        self.video_source = DEFAULT_VIDEO_PATH if response else video_path
        self.cap = cv2.VideoCapture(self.video_source)
        self.predicting = False
        HEIGHT, WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        # Create video canvas
        self.video_canvas = ctk.CTkCanvas(root, width=WIDTH, height=HEIGHT, bg="white")
        self.video_canvas.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.fig, self.ax = plt.subplots(figsize=(4, 8), dpi=100)
        self.fig.tight_layout(pad=1.4)
        self.ax.imshow(
            cv2.cvtColor(cv2.imread(BACKGROUND_IMAGE_PATH), cv2.COLOR_BGR2RGB),
            extent=[0, 8.5, 0, 32],
        )
        self.set_axes()
        self.people = None
        self.predictions = None

        # Create canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().configure(bg="white")
        self.canvas.get_tk_widget().grid(row=0, column=2, padx=1, pady=1, rowspan=6)
        self.canvas.draw_idle()

        # Create buttons and radio buttons
        self.pause_button = ctk.CTkButton(root, text="Pause", command=self.toggle_pause)
        self.pause_button.grid(row=1, column=1, pady=10, padx=15, sticky=tk.W)
        self.paused = False

        self.predict_button = ctk.CTkButton(
            root, text="Predict!", command=self.toggle_predict
        )
        self.predict_button.grid(row=1, column=0, pady=10, padx=15, sticky=tk.E)

        options = ["Predict 1s", "Predict 3s", "Predict 5s"]
        self.selected_option = tk.StringVar(value=options[0])
        for i, option in enumerate(options):
            radio_button = ctk.CTkRadioButton(
                root, text=option, variable=self.selected_option, value=option
            )
            radio_button.grid(row=i + 3, column=0, sticky=tk.W, pady=7, padx=15)

        self.selected_option.trace("w", lambda *args: self.handle_radio_button_change())

        self.n_steps = int(self.selected_option.get().split()[1][0])

        self.tracker = Tracker(self.cap, draw_tracking_lines=False)
        self.track = self.tracker.track()

        self.predictor = Predictor(
            predicting_model_path if predicting_model_path else DEFAULT_MODEL_PATH,
        )
        self.colors = []

        self.update()

    def set_axes(self):
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_xticks(np.arange(0, 8.1, 1))
        self.ax.set_yticks(
            np.arange(0, 32.1, 1), labels=[None if i % 2 else i for i in range(33)]
        )
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(32, 0)
        self.ax.grid()

    def clear_plot(self):
        self.ax.cla()
        self.ax.imshow(
            cv2.cvtColor(cv2.imread(BACKGROUND_IMAGE_PATH), cv2.COLOR_BGR2RGB),
            extent=[0, 8.5, 0, 32],
        )
        self.set_axes()

    def plot_people_on_plan(
        self,
        people,
        predictions=None,
        plot_actual=True,
        plot_predictions=True,
        clear=True,
    ):
        def color(track_id):
            if track_id not in self.colors:
                self.colors.append(track_id)
            return self.colors.index(track_id)

        if clear:
            self.clear_plot()

        if people is not None:
            for track_id, (x, y) in people.items():
                if plot_actual:
                    self.people = plt.scatter(
                        x,
                        y,
                        label=f"Person {track_id}",
                        marker="o",
                        color=f"C{color(track_id)}",
                    )
                if (
                    plot_predictions
                    and predictions is not None
                    and track_id in predictions
                ):
                    plt.plot(
                        predictions[track_id][:10, 0],
                        predictions[track_id][:10, 1],
                        label=f"Person {track_id} prediction",
                        color=f"C{color(track_id)}",
                    )

    def update(self):
        if self.paused:
            self.root.after(50, self.update)
            return
        try:
            if not self.predicting:
                frame, people, track_hist = next(self.track)

                self.photo = self.convert_to_tk_image(frame)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                self.plot_people_on_plan(people, plot_predictions=False)
                self.canvas.draw_idle()
            elif self.iteration_counter > 0:
                frame, people, track_hist = next(self.track)
                self.stored_frames.append(frame)

                self.predictor.update(track_hist)
                predictions = self.predictor.predict()

                self.plot_people_on_plan(
                    people, predictions, plot_actual=False, clear=False
                )
                self.canvas.draw_idle()

                self.iteration_counter -= 1

            else:
                if self.stored_frames:
                    frame = self.stored_frames.pop(0)
                    self.photo = self.convert_to_tk_image(frame)
                    self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                    self.canvas.draw_idle()
                    time.sleep(0.1)
                else:
                    self.canvas.draw_idle()
                    self.toggle_predict()

        except StopIteration:
            self.handle_stop_iteration()
            return

        self.root.after(50, self.update)

    def handle_stop_iteration(self):
        self.cap.release()
        self.root.after_cancel(self.update)
        tk.messagebox.showinfo("Video ended", f"Video ended")

    def convert_to_tk_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img)

    def toggle_predict(self):
        self.predicting = not self.predicting
        self.iteration_counter = self.n_steps * 10 if self.predicting else 0
        self.stored_frames = []

    def toggle_pause(self):
        self.paused = not self.paused

    def handle_radio_button_change(self):
        self.n_steps = int(self.selected_option.get().split()[1][0])

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()

    def on_closing(self):
        response = tk.messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if response:
            self.paused = True
            self.root.quit()

    @staticmethod
    def run(predicting_model_path):
        ctk.set_appearance_mode("light")
        root = tk.Tk()
        app = PedestrianTrajectoryPredictionApp(root, predicting_model_path)
        app.root.protocol("WM_DELETE_WINDOW", app.on_closing)

        root.mainloop()
