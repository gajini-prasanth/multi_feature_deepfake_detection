import argparse
import subprocess
import sys

from evaluation.ablation import profile_ablation_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Feature Deepfake & AI Image Detection")
    parser.add_argument("command", choices=["profile", "app"], help="Action to run")
    args = parser.parse_args()

    if args.command == "profile":
        df = profile_ablation_models()
        print(df.to_string(index=False))
    elif args.command == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], check=True)


if __name__ == "__main__":
    main()
