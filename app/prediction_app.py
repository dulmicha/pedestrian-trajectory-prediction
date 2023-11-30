import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
from .components.predict import Predictor
import time


class PedestrianTrajectoryPredictionApp:
    def __init__(self, root, video_path=None, predicting_model_path=None):
        self.root = root
        self.root.title("Pedestrian trajectory prediction")
        self.root.resizable(False, False)

        # Variables for video and plotting
        self.video_source = (
            "data/default_video.mp4" if video_path is None else video_path
        )
        self.cap = cv2.VideoCapture(self.video_source)
        self.paused = False
        HEIGHT, WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        # Create video canvas
        self.video_canvas = ctk.CTkCanvas(root, width=WIDTH, height=HEIGHT, bg="white")
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
        self.people = None
        self.predictions = None

        # Create canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().configure(bg="white")
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        self.canvas.draw_idle()

        # Create buttons and radio buttons
        self.pause_button = ctk.CTkButton(
            root, text="Predict!", command=self.toggle_pause
        )
        self.pause_button.grid(row=1, column=0, pady=10)

        options = ["Predict 1s", "Predict 3s", "Predict 5s"]
        self.selected_option = tk.StringVar(value=options[0])
        for i, option in enumerate(options):
            radio_button = ctk.CTkRadioButton(
                root, text=option, variable=self.selected_option, value=option
            )
            radio_button.grid(row=i + 2, column=0, sticky=tk.W, pady=5, padx=10)

        self.selected_option.trace("w", lambda *args: self.handle_radio_button_change())

        self.n_steps = int(self.selected_option.get().split()[1][0])

        self.predictor = Predictor(
            predicting_model_path
            if predicting_model_path
            else "models/model_td_100lstm.h5",
            self.cap,
        )
        self.prediction = self.predictor.predict()
        self.colors = []

        self.update()

    def plot_people_on_plan(
        self,
        people,
        i,
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
            self.ax.cla()
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
                    (self.predictions,) = plt.plot(
                        predictions[track_id][:10, 0],
                        predictions[track_id][:10, 1],
                        label=f"Person {track_id} prediction",
                        color=f"C{color(track_id)}",
                    )

    def update(self):
        try:
            if not self.paused:
                frame, *people = next(self.prediction)

                self.photo = self.convert_to_tk_image(frame)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

                self.plot_people_on_plan(*people, plot_predictions=False)
                self.canvas.draw_idle()
            else:
                frame, *people = next(self.prediction)
                self.stored_frames.append(frame)

                self.plot_people_on_plan(*people, plot_actual=False, clear=False)
                self.canvas.draw_idle()

                if self.iteration_counter == 0:
                    for frame in self.stored_frames:
                        self.photo = self.convert_to_tk_image(frame)
                        self.video_canvas.create_image(
                            0, 0, anchor=tk.NW, image=self.photo
                        )

                    self.plot_people_on_plan(
                        *people, plot_predictions=False, clear=False
                    )
                    self.canvas.draw_idle()

                    self.toggle_pause()

                self.iteration_counter -= 1

        except StopIteration:
            self.handle_stop_iteration()
            return

        self.root.after(20, self.update)

    def handle_stop_iteration(self):
        self.predictor.tracker.stop = True
        self.cap.release()
        self.root.after_cancel(self.update)
        tk.messagebox.showinfo("Video ended", f"Video ended")

    def convert_to_tk_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img)

    def toggle_pause(self):
        self.paused = not self.paused
        self.iteration_counter = self.n_steps
        self.stored_frames = []

    def handle_radio_button_change(self):
        self.n_steps = int(self.selected_option.get().split()[1][0])

    def __del__(self):
        if hasattr(self, "cap"):
            self.predictor.tracker.stop = True
            self.cap.release()

    def on_closing(self):
        response = tk.messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if response:
            self.root.after_cancel(self.update)
            self.root.withdraw()
            self.root.quit()

    @staticmethod
    def run(video_path, predicting_model_path):
        ctk.set_appearance_mode("light")
        root = tk.Tk()
        app = PedestrianTrajectoryPredictionApp(root, video_path, predicting_model_path)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)

        root.mainloop()
