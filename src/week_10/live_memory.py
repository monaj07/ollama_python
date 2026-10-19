"""Persist actual live camera attempts independently of conversational summaries."""
from __future__ import annotations

from datetime import datetime, timezone

from memory_store import add_observation


class RecordingTools:
    """Wrap the existing dispatcher; save each camera call exactly once."""

    def __init__(self, tools, conversation_session_id: str):
        self.tools = tools
        self.conversation_session_id = conversation_session_id
        self.pir = None

    def begin_turn(self) -> None:
        self.pir = None

    def close(self) -> None:
        self.tools.close()

    def execute(self, name: str, args: dict) -> dict:
        try:
            result = self.tools.execute(name, args)
        except Exception as exc:
            if name == "observe_scene":
                self._save({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, args)
            raise
        if name == "detect_motion":
            self.pir = result if isinstance(result, dict) else None
        elif name == "observe_scene":
            self._save(result if isinstance(result, dict) else {}, args)
        return result

    def _save(self, camera: dict, args: dict) -> None:
        counts = camera.get("counts")
        count = counts.get("person") if isinstance(counts, dict) else None
        person_known = camera.get("ok") is True and type(count) is int and count >= 0
        detections = camera.get("detections")
        detections = detections if isinstance(detections, list) else []
        scores = [
            item["confidence"] for item in detections
            if isinstance(item, dict) and item.get("label") == "person"
            and type(item.get("confidence")) in (int, float)
            and 0 <= item["confidence"] <= 1
        ]
        score = max(scores, default=None) if person_known and count > 0 else None
        motion = (self.pir.get("motion_signal_active")
                  if self.pir and self.pir.get("ok") is True else None)
        if type(motion) is not bool:
            motion = None

        stamp = camera.get("observed_at")
        try:
            parsed = datetime.fromisoformat(stamp)
            if parsed.utcoffset() is None:
                raise ValueError("Missing camera timezone")
        except (TypeError, ValueError):
            stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            timestamp_source = "tool_completion"
        else:
            timestamp_source = "camera_receipt"

        evidence = camera.get("source")
        event_id = add_observation(
            trigger="live_observation",
            observed_at_utc=stamp,
            person_detected=count > 0 if person_known else None,
            detection_score=score,
            evidence_path=evidence if isinstance(evidence, str) else None,
            details={
                "conversation_session_id": self.conversation_session_id,
                "targets": args.get("targets", []),
                "camera_ok": camera.get("ok") is True,
                "person_count": count if person_known else None,
                "pir_signal_active": motion,
                "pir_observed_at_utc": self.pir.get("observed_at") if self.pir else None,
                "camera_observed_at_utc": camera.get("observed_at"),
                "timestamp_source": timestamp_source,
                "vision_elapsed_s": camera.get("tool_elapsed_s"),
                "confidence_threshold": camera.get("person_count_threshold"),
                "detections": detections,
                "vision_error": camera.get("error"),
            },
        )
        print(f"[observation] saved event={event_id} trigger=live_observation", flush=True)
