# Provenance of recorded experiment outputs

- `photos/before.jpg` and `photos/after.jpg`: **original photographs** from the user, at 640×480. The mug moves from left to right, the glass lidded bowl stays still; the fixed camera views indoor furniture and moving foliage through the window.
- `photos/*_320.jpg`, `photos/*_480.jpg`: **reconstructed downsampled copies**, Pillow Lanczos and JPEG quality 95. The originals were resized for the VLM/YOLO experiments, but the precise original JPEG save parameters weren't recorded. These recreations are not claimed to be byte-identical to benchmark input copies.
- `yolo_ncnn/320/{before,after}/*`: **unaltered raw** `metrics.json`, `observation.json`, and `annotated.jpg` from the supplied 320-model ZIP.
- `yolo_ncnn/640/before/*`: **unaltered raw** 640-model before-image outputs supplied separately. No 640-model after-image observation file was provided.
- `change_detection/pixel_diff_320/*`: exact 320×240 diff/mask images supplied by the user, from an initial hand-set grayscale threshold of 30.
- `change_detection/otsu_user_320/*`: exact 320×240 Otsu rerun output images supplied by the user. These should not be confused with the 640×480 measured results.
- `change_detection/otsu_full_640/{mask.png,regions.png,metrics.json}`: **regenerated** against the supplied originals using the committed OpenCV method. Threshold 41, 3.68% changed pixels, two largest connected components. The `summary.png` is the previously generated four-panel figure supplied in the conversation.
- `vlm_benchmarks.csv`: exact *reported* server counters; **qualitative answer summaries**, not verbatim model logs.
- `vlm_response_notes.md`: limitations and known answer interpretations.

The photos show a private indoor scene. Review `data/photos/` before pushing to a public repository if privacy is a concern.
