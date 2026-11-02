"""Full-frame change detection with Otsu; no manually selected ROI.

Run from repository root:
    python src/week_14/experiments/change_detection_otsu.py \
        src/week_14/data/photos/before.jpg src/week_14/data/photos/after.jpg \
        --output-dir /tmp/week14_otsu

Requires OpenCV and NumPy. Output is based on the two input frames; the two
largest components are *candidate changed regions*, not recognised objects.
"""
import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    before = cv2.imread(str(args.before))
    after = cv2.imread(str(args.after))
    if before is None or after is None or before.shape != after.shape:
        raise ValueError('Both images must exist and have identical dimensions')

    before = cv2.GaussianBlur(before, (3, 3), 0)
    after_smooth = cv2.GaussianBlur(after, (3, 3), 0)
    diff = cv2.cvtColor(cv2.absdiff(before, after_smooth), cv2.COLOR_BGR2GRAY)
    threshold, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = sorted(stats[1:], key=lambda s: s[cv2.CC_STAT_AREA], reverse=True)[:2]

    overlay = after.copy()
    for rank, (x, y, w, h, area) in enumerate(regions, 1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 255), 2)
        cv2.putText(overlay, f'Region {rank}', (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 205), 2, cv2.LINE_AA)
        print(f'Region {rank}: xywh=({x},{y},{w},{h}), area={area} pixels')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / 'mask.png'), mask)
    cv2.imwrite(str(args.output_dir / 'regions.png'), overlay)
    print(f'Otsu threshold: {threshold:.0f}/255')
    print(f'Changed pixels: {(mask > 0).mean() * 100:.2f}%')
    print(f'Images saved to {args.output_dir}')


if __name__ == '__main__':
    main()
