"""Contract and failure-path tests; no camera, model downloads or Ollama required."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest
import venv
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/week_07_08"))
from vision_observation import validate_observation
from yoloe_detector import YOLOEDetector
from observe_scene import SceneObserver
from reason_yoloe import summarize_observation
import detect as fixed_detector

NOW = datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)


def observation(now=NOW):
    return {
        "schema_version": "pi-vision-v1",
        "targets": ["glasses", "cup"],
        "counts": {"glasses": 1, "cup": 0},
        "confidence_threshold": 0.2,
        "received_at_utc": now.isoformat(),
        "detections": [{"label": "glasses", "confidence": 0.21,
                        "bbox_xyxy": [10, 10, 40, 40]}],
    }


class ObservationTests(unittest.TestCase):
    def test_valid_payload_preserves_zero_and_scores(self):
        result = validate_observation(observation(), now=NOW)
        self.assertEqual(result["counts"]["cup"], 0)
        self.assertEqual(result["scores_by_label"]["glasses"], [0.21])

    def test_stale_future_missing_and_naive_timestamps_rejected(self):
        for stamp in [(NOW - timedelta(seconds=61)).isoformat(),
                      (NOW + timedelta(seconds=3)).isoformat(), None,
                      NOW.replace(tzinfo=None).isoformat()]:
            with self.subTest(stamp=stamp):
                item = observation()
                item["received_at_utc"] = stamp
                with self.assertRaises(ValueError):
                    validate_observation(item, now=NOW)

    def test_count_mismatch_rejected(self):
        item = observation()
        item["counts"]["glasses"] = 2
        with self.assertRaises(ValueError):
            validate_observation(item, now=NOW)

    def test_invalid_values_rejected(self):
        changes = [
            ("counts", {"glasses": True, "cup": 0}),
            ("counts", {"glasses": 1}),
            ("confidence_threshold", float("nan")),
            ("targets", ["glasses", "GLASSES"]),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                item = observation()
                item[field] = value
                with self.assertRaises(ValueError):
                    validate_observation(item, now=NOW)
        for max_age in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                validate_observation(observation(), max_age=max_age, now=NOW)

    def test_below_threshold_foreign_label_and_bad_box_rejected(self):
        for changes in [{"confidence": 0.1}, {"label": "person"},
                        {"bbox_xyxy": [40, 10, 10, 40]},
                        {"bbox_xyxy": [0, 0, float("inf"), 10]}]:
            item = observation()
            item["detections"][0].update(changes)
            with self.assertRaises(ValueError):
                validate_observation(item, now=NOW)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Vector:
    def tolist(self):
        return [10, 10, 40, 40]


class FakeYOLOE:
    instances = []

    def __init__(self, model):
        self.classes = []
        self.set_calls = []
        self.predict_calls = []
        self.predictor = SimpleNamespace(imgsz=[640, 640])
        self.instances.append(self)

    def set_classes(self, targets):
        self.classes = list(targets)
        self.set_calls.append(list(targets))

    def predict(self, image, **kwargs):
        self.predict_calls.append(kwargs)
        return [SimpleNamespace(
            names=dict(enumerate(self.classes)),
            boxes=[SimpleNamespace(cls=Scalar(0), conf=Scalar(0.4), xyxy=[Vector()])],
            plot=lambda **kwargs: b"annotated",
        )]


def fake_write(path, image):
    Path(path).write_bytes(b"annotated")
    return True


FAKE_MODULES = {
    "ultralytics": SimpleNamespace(YOLOE=FakeYOLOE),
    "cv2": SimpleNamespace(imread=lambda path: SimpleNamespace(shape=(480, 640, 3)),
                           imwrite=fake_write),
}


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "scene.jpg"
        self.source.write_bytes(b"fake fixture")
        self.source.with_suffix(".json").write_text(json.dumps({"received_at_utc": NOW.isoformat()}))
        FakeYOLOE.instances.clear()
        self.patch = patch.dict(sys.modules, FAKE_MODULES)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_reuses_model_and_prompts_and_preserves_capture_time(self):
        detector = YOLOEDetector()
        first = detector.detect(self.source, ["cup", "glasses"])
        detector.detect(self.source, ["cup", "glasses"])
        changed = detector.detect(self.source, ["shoe"])
        self.assertEqual(len(FakeYOLOE.instances), 1)
        self.assertEqual(detector.model.set_calls, [["cup", "glasses"], ["shoe"]])
        self.assertEqual(changed["counts"], {"shoe": 1})
        self.assertEqual(first["received_at_utc"], NOW.isoformat())
        self.assertFalse(detector.model.predict_calls[0]["rect"])
        with self.assertRaises(ValueError):
            validate_observation(first, now=NOW + timedelta(minutes=2))

    def test_benchmark_is_opt_in_and_outputs_keep_true_input_size(self):
        detector = YOLOEDetector()
        outdir = Path(self.temp.name) / "results"
        result = detector.detect(self.source, ["cup"], runs=3, outdir=outdir)
        self.assertEqual(len(detector.model.predict_calls), 4)
        self.assertEqual(result["metrics"]["warm_runs"], 3)
        self.assertEqual(result["predictor_image_size"], [640, 640])
        self.assertTrue((outdir / "observation.json").is_file())
        self.assertTrue((outdir / "annotated.jpg").is_file())

    def test_missing_metadata_stays_missing(self):
        self.source.with_suffix(".json").unlink()
        result = YOLOEDetector().detect(self.source, ["cup"])
        self.assertIsNone(result["received_at_utc"])
        with self.assertRaises(ValueError):
            validate_observation(result)

    def test_capture_prepares_targets_first_and_has_unique_paths(self):
        script = Path(self.temp.name) / "camera_lab.py"
        script.touch()
        observer = SceneObserver(camera_python=sys.executable, camera_script=script,
                                 outdir=Path(self.temp.name) / "captures")

        def capture(command, **kwargs):
            self.assertEqual(observer.detector.model.classes, ["glasses"])
            source = Path(command[-1])
            source.write_bytes(b"captured")
            source.with_suffix(".json").write_text(json.dumps({
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
            }))

        with patch("observe_scene.subprocess.run", side_effect=capture):
            first = observer.observe_scene(["glasses"])
            second = observer.observe_scene(["glasses"])
        self.assertNotEqual(first["source"], second["source"])
        self.assertEqual(len(observer.detector.model.set_calls), 1)

    def test_camera_failure_stops_before_prediction(self):
        script = Path(self.temp.name) / "camera_lab.py"
        script.touch()
        observer = SceneObserver(camera_python=sys.executable, camera_script=script,
                                 outdir=Path(self.temp.name) / "captures")
        with patch("observe_scene.subprocess.run", side_effect=subprocess.CalledProcessError(
                1, ["camera"], stderr="Camera unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Camera unavailable"):
                observer.observe_scene(["glasses"])
        self.assertEqual(observer.detector.model.predict_calls, [])

    def test_camera_python_symlink_keeps_its_virtual_environment(self):
        env_dir = Path(self.temp.name) / "camera-env"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
        env_python = env_dir / "bin/python"
        script = Path(self.temp.name) / "camera_lab.py"
        script.touch()
        observer = SceneObserver(camera_python=env_python, camera_script=script)
        self.assertEqual(observer.camera_python, env_python.absolute())
        result = subprocess.run(
            [str(observer.camera_python), "-c", "import sys; print(sys.prefix)"],
            check=True, text=True, capture_output=True,
        )
        self.assertEqual(Path(result.stdout.strip()), env_dir)

    def test_camera_timeout_exposes_byte_logs_without_running_detection(self):
        script = Path(self.temp.name) / "camera_lab.py"
        script.touch()
        observer = SceneObserver(camera_python=sys.executable, camera_script=script,
                                 outdir=Path(self.temp.name) / "captures")
        error = subprocess.TimeoutExpired(["camera"], 60,
                                          output=b"Camera started\n", stderr=b"Waiting for frame\n")
        with patch("observe_scene.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Camera capture timed out") as raised:
                observer.observe_scene(["glasses"])
        self.assertIn("Camera started", str(raised.exception))
        self.assertIn("Waiting for frame", str(raised.exception))
        self.assertEqual(observer.detector.model.predict_calls, [])

    def test_legacy_detector_records_requested_and_actual_input_sizes(self):
        class FixedSizeYOLO(FakeYOLOE):
            def __init__(self, model, task=None):
                super().__init__(model)
                self.classes = ["cup"]
                self.predictor.imgsz = [320, 320]

        outdir = Path(self.temp.name) / "fixed-size"
        modules = {
            "ultralytics": SimpleNamespace(YOLO=FixedSizeYOLO),
            "numpy": SimpleNamespace(median=statistics.median,
                                     percentile=lambda values, q: values[0]),
        }
        args = ["detect.py", "--source", str(self.source), "--imgsz", "640",
                "--runs", "1", "--outdir", str(outdir)]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", args), patch("sys.stdout", new_callable=io.StringIO):
            fixed_detector.main()
        for filename in ["observation.json", "metrics.json"]:
            record = json.loads((outdir / filename).read_text())
            self.assertEqual(record["input_size_requested"], 640)
            self.assertEqual(record["predictor_image_size"], [320, 320])
            self.assertNotIn("input_size", record)


class ReasonTests(unittest.TestCase):
    def test_stale_observation_never_calls_ollama(self):
        with patch("reason_yoloe.urllib.request.urlopen") as http:
            with self.assertRaises(ValueError):
                summarize_observation(observation(datetime.now(timezone.utc) - timedelta(minutes=5)))
            http.assert_not_called()

    def test_one_json_request_with_scores_no_image_and_thinking_disabled(self):
        reply = {"message": {"content": "The detector reported one possible pair of glasses."},
                 "done": True, "done_reason": "stop"}
        with patch("reason_yoloe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as http:
            result = summarize_observation(observation(datetime.now(timezone.utc)))
            self.assertEqual(http.call_count, 1)
            body = json.loads(http.call_args.args[0].data)
            self.assertFalse(body["think"])
            self.assertFalse(body["stream"])
            self.assertNotIn("images", body["messages"][1])
            data = json.loads(body["messages"][1]["content"])
            self.assertEqual(data["scores_by_label"]["glasses"], [0.21])
            self.assertFalse(result["stale_at_response"])

    def test_empty_and_truncated_responses_rejected(self):
        for reply in [{"message": {"content": " "}, "done": True},
                      {"message": {"content": "partial"}, "done": True, "done_reason": "length"}]:
            with patch("reason_yoloe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())):
                with self.assertRaises(ValueError):
                    summarize_observation(observation(datetime.now(timezone.utc)))

    def test_zero_count_labels_are_removed_from_entire_llm_input(self):
        item = observation(datetime.now(timezone.utc))
        item["targets"] = ["cup", "teapot", "glasses", "shoe"]
        item["counts"] = {"cup": 3, "teapot": 0, "glasses": 0, "shoe": 0}
        item["detections"] = [
            {"label": "cup", "confidence": score, "bbox_xyxy": [10, 10, 40, 40]}
            for score in [0.3, 0.4, 0.5]
        ]
        reply = {"message": {"content": "The detector reported three possible cups."},
                 "done": True, "done_reason": "stop"}
        with patch("reason_yoloe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as http:
            record = summarize_observation(item)
        body = json.loads(http.call_args.args[0].data)
        sent = json.dumps(body["messages"])
        for omitted in ["teapot", "glasses", "shoe"]:
            self.assertNotIn(omitted, sent)
        self.assertEqual(record["input"]["counts"], {"cup": 3})
        self.assertEqual(record["validated_observation"]["counts"], item["counts"])
        self.assertEqual(record["input"]["scores_by_label"], {"cup": [0.3, 0.4, 0.5]})

    def test_all_zero_counts_send_no_target_names(self):
        item = observation(datetime.now(timezone.utc))
        item["counts"] = {"glasses": 0, "cup": 0}
        item["detections"] = []
        reply = {"message": {"content": "The detector reported no detections above the configured threshold."},
                 "done": True, "done_reason": "stop"}
        with patch("reason_yoloe.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as http:
            record = summarize_observation(item)
        body = json.loads(http.call_args.args[0].data)
        sent = json.dumps(body["messages"])
        self.assertNotIn("glasses", sent)
        self.assertNotIn("cup", sent)
        self.assertEqual(record["input"]["counts"], {})
        self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
