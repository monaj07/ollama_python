"""Generate 320x240 and 480x360 copies of the original 640x480 images.

These are *reconstructed* JPEGs. Quality settings or Pillow versions may
produce files different from those used in the original Pi measurements.
"""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in (args.before, args.after):
        source = Image.open(input_path).convert('RGB')
        for width, height in ((320, 240), (480, 360)):
            destination = args.output_dir / f'{input_path.stem}_{width}.jpg'
            source.resize((width, height), Image.Resampling.LANCZOS).save(
                destination, quality=95, subsampling=0
            )
            print(destination)


if __name__ == '__main__':
    main()
