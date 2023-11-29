import argparse
from app.prediction_app import PedestrianTrajectoryPredictionApp


def print_usage():
    print(
        "Usage: python main.py [--debug] --video <path_to_video> --model <path_to_predicting_model>"
    )
    print("Example: python main.py --video data/video.mp4 --model models/model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Run app in debug mode", action="store_true")
    parser.add_argument("--video", help="Path to video", required=False)
    parser.add_argument("--model", help="Path to predicting model", required=False)
    args = parser.parse_args()
    video_path, model_path = args.video, args.model
    if not args.debug:
        if video_path is None or model_path is None:
            print_usage()
            exit(1)

    PedestrianTrajectoryPredictionApp.run(video_path, model_path)
