"""Hardware-independent regressions for Week 09 persistent vision bridge."""
from __future__ import annotations

import io
import json
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from tools import PersistentVisionWorker, PhysicalTools
from vision_bridge import compact_observation, serve


def test_filter_preserves_raw_person_counts_and_other_classes():
    original = {
        "detections": [
            {"label": "person", "confidence": .22},
            {"label": "person", "confidence": .82},
            {"label": "cup", "confidence": .25},
        ],
        "targets": ["person", "cup"],
        "counts": {"person": 2, "cup": 1},
        "confidence_threshold": 0.2,
        "received_at_utc": "2026-10-17T08:25:00+00:00",
        "source": "outputs/observations/scene.jpg",
    }
    result = compact_observation(original)
    assert result["counts"] == {"person": 1, "cup": 1}
    assert result["raw_counts"] == {"person": 2, "cup": 1}
    assert result["person_count_threshold"] == .3
    assert [x["label"] for x in result["detections"]] == ["person", "cup"]


def test_bridge_handles_multiple_requests_with_same_observer(monkeypatch):
    class FakeObserver:
        created = 0
        def __init__(self):
            self.__class__.created += 1
        def observe_scene(self, targets, conf):
            return {
                "detections": [{"label": targets[0], "confidence": .75}],
                "targets": targets,
                "counts": {targets[0]: 1},
                "confidence_threshold": conf,
                "received_at_utc": "test",
                "source": "x.jpg",
            }

    import types
    monkeypatch.setitem(sys.modules, "observe_scene", types.SimpleNamespace(SceneObserver=FakeObserver))
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        '{"targets":["person"],"conf":0.2}\n'
        '{"targets":["person"],"conf":0.2}\n'
    ))
    capture = io.StringIO()
    monkeypatch.setattr(sys, "stdout", capture)
    serve(persistent=True)
    replies = [json.loads(line) for line in capture.getvalue().splitlines()]
    assert FakeObserver.created == 1
    assert len(replies) == 2
    assert all(r["counts"] == {"person": 1} for r in replies)


def write_fake_worker(tmp_path, contents):
    worker = tmp_path / "fake_bridge.py"
    worker.write_text(textwrap.dedent(contents))
    return worker


def test_process_reuse_and_recovery_after_crash(tmp_path):
    worker = write_fake_worker(tmp_path, """
        import json, os, sys
        for line in sys.stdin:
            req = json.loads(line)
            print(json.dumps({'ok': True, 'pid': os.getpid(), 'targets': req['targets']}), flush=True)
    """)
    p = PersistentVisionWorker(Path(sys.executable), worker, timeout=3)
    try:
        a = p.request({"targets": ["person"], "conf": .2})
        b = p.request({"targets": ["cup"], "conf": .2})
        assert a["ok"] and b["ok"]
        assert a["pid"] == b["pid"]
        assert b["targets"] == ["cup"]
        p.process.kill()
        p.process.wait()
        c = p.request({"targets": ["person"], "conf": .2})
        assert c["ok"] and c["pid"] != a["pid"]
    finally:
        p.close()


def test_worker_timeout_returns_error_and_can_recover(tmp_path):
    worker = write_fake_worker(tmp_path, """
        import json, sys, time
        for line in sys.stdin:
            req = json.loads(line)
            if req.get('sleep'):
                time.sleep(10)
            print(json.dumps({'ok': True}), flush=True)
    """)
    p = PersistentVisionWorker(Path(sys.executable), worker, timeout=1.0)
    try:
        failure = p.request({"sleep": True})
        assert failure["ok"] is False
        assert "timed out" in failure["error"]
        success = p.request({"sleep": False})
        assert success["ok"] is True
    finally:
        p.close()


def test_worker_invalid_response_recovers(tmp_path):
    worker = write_fake_worker(tmp_path, """
        import json, sys
        for line in sys.stdin:
            req = json.loads(line)
            print('invalid' if req.get('bad') else json.dumps({'ok': True}), flush=True)
    """)
    p = PersistentVisionWorker(Path(sys.executable), worker, timeout=3)
    try:
        bad = p.request({"bad": True})
        assert not bad["ok"]
        assert p.request({"bad": False})["ok"]
    finally:
        p.close()


def test_physical_tools_still_validates_call(monkeypatch):
    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.calls = []
        def request(self, args):
            self.calls.append(args)
            return {"ok": True, "counts": {"person": 0}}
        def close(self):
            pass
    monkeypatch.setattr("tools.PersistentVisionWorker", FakeWorker)
    t = PhysicalTools()
    try:
        answer = t.execute("observe_scene", {"targets": ["person"]})
        assert answer["ok"] is True
        assert t._vision_worker.calls == [{"targets": ["person"], "conf": .2}]
        with pytest.raises(ValueError):
            t.execute("observe_scene", {"targets": ["person"], "conf": 100})
    finally:
        t.close()
