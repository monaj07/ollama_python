"""Initial hand-thresholded difference baseline (grayscale difference > 30)."""
import argparse
from pathlib import Path
from PIL import Image, ImageChops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--threshold', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    before = Image.open(args.before).convert('RGB')
    after = Image.open(args.after).convert('RGB')
    if before.size != after.size:
        raise ValueError('Input images must have the same dimensions')
    diff = ImageChops.difference(before, after)
    mask = diff.convert('L').point(lambda x: 255 if x > args.threshold else 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diff.save(args.output_dir / 'diff.png')
    mask.save(args.output_dir / 'mask.png')
    print(f'Saved {args.output_dir / "diff.png"} and {args.output_dir / "mask.png"}')


if __name__ == '__main__':
    main()
