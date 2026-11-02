# VLM responses — preserved observations, not verbatim transcripts

These six experiments were reported in the conversation while the Raspberry Pi
ran the local models. The original terminal outputs were **not uploaded**, so
this file intentionally does not invent exact model responses or client timings.
Reported server-side timings and token counts are in [`vlm_benchmarks.csv`](vlm_benchmarks.csv).

- **LFM3-VL:2b, two 640×480 photos:** wrongly claimed the mug was replaced by a bowl and mentioned an additional bowl.
- **LFM3-VL:2b, one 640×480 after photo:** correctly identified the patterned mug's position relative to the glass bowl; generated candidate normalized coordinates `[822,571,925,738]` (mug) and `[519,512,663,701]` (bowl), **not validated against pixel ground truth**.
- **LFM3-VL:2b, two 480×360 photos:** partially identified the mug on the right but incorrectly stated the glass ornament had moved or swapped positions.
- **LFM3-VL:2b, two 320×240 photos:** correctly identified the mug moving left to right; added an unsupported assertion about orange colour.
- **Gemma4:e2b, two 320×240 photos:** incorrectly said the ornament disappeared.
- **Gemma4:e2b, one 320×240 after photo:** correctly described mug position relative to the glass bowl.

The comparison prompt is preserved exactly in `experiments/compare_vlm.py`.
The earlier one-image prompt text is only reconstructed, not original verbatim.
No multi-seed or multi-image dataset evaluation was done; do not treat a single
successful run as an accuracy estimate.
