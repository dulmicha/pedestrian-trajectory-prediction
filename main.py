import argparse
from app.prediction_app import PedestrianTrajectoryPredictionApp


def print_usage():
    print("Usage: python main.py [--model <path_to_predicting_model>]")
    print("Example: python main.py --model models/model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to predicting model", required=False)
    args = parser.parse_args()
    model_path = args.model if args.model else None

    PedestrianTrajectoryPredictionApp.run(model_path)
