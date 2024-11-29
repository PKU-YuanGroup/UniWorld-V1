import sys
import argparse
from pathlib import Path
sys.path.append(".")
from causalvideovae.model import *

def main():
    args = parse_args()

    try:
        model_cls = ModelRegistry.get_model(args.model_name)
    except KeyError:
        print(f"Error: Model '{args.model_name}' not found in ModelRegistry.")
        return

    try:
        vae = model_cls.from_pretrained(args.from_pretrained)
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        vae.save_pretrained(str(output_path))
        print(f"Model successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Load and save a pretrained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
    parser.add_argument("--from_pretrained", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the model")
    return parser.parse_args()

if __name__ == "__main__":
    main()