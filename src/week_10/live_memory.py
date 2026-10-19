"""Record live sensor results without coupling hardware success to SQLite success.

The camera response stays authoritative for the current turn. SQLite is the
source of truth for any *subsequent* question about previously saved readings.
"""
from __future__ import annotations

from datetime import datetime, timezone

from memory_store import add_observation

def render_recorded_observation(observation: dict | None) -> str:
    """Summarize an earlier stored sensor event, never implying a fresh check."""
    if observation is None:
        return ("I haven't taken a new reading. I couldn't find a saved live "
                "observation for this conversation.")

    when = observation["time_local"]
    count = observation["person_count"]
    motion = observation["pir_signal_active"]
    if observation["camera_ok"] is not True or count is None:
        camera_clause = "the camera result was unavailable"
    elif count == 0:
        camera_clause = "the camera did not detect a person in that frame"
    else:
        people = "person" if count == 1 else "people"
        camera_clause = f"the camera reported {count} {people}"
        score = observation["detection_score"]
        if score is not None:
            camera_clause += f" (highest detector score {score:.3f})"

    pir_clause = (
        "the PIR reported movement" if motion is True else
        "the PIR reported no movement" if motion is False else
        "the PIR reading was unavailable"
    )
    return (f"I haven't taken a new reading. At {when}, {camera_clause}; "
            f"{pir_clause}. That was one earlier observation, not a check "
            "of the room right now.")


class RecordingTools:
    """Wrap the existing dispatcher and *attempt* to persist each camera call.

    Saving is best effort: database failures never replace a sensor response or
    hide the original sensor exception. Inspect ``last_recording`` for status.
    """

    def __init__(self, tools, conversation_session_id: str):
        self.tools = tools
        self.conversation_session_id = conversation_session_id
        self.pir: dict | None = None
        self.last_recording: dict | None = None

    def begin_turn(self) -> None:
        self.pir = None
        self.last_recording = None

    def close(self) -> None:
        self.tools.close()

    def execute(self, name: str, args: dict) -> dict:
        try:
            result = self.tools.execute(name, args)
        except Exception as exc:
            if name == "observe_scene":
                self._try_save({"ok": False,
                                "error": f"{type(exc).__name__}: {exc}"}, args)
            raise  # Preserve the original hardware/vision exception.

        if name == "detect_motion":
            self.pir = result if isinstance(result, dict) else None
        elif name == "observe_scene":
            self._try_save(result if isinstance(result, dict) else {}, args)
        return result  # Never replace valid hardware evidence with a DB error.

    def _try_save(self, camera: dict, args: dict) -> None:
        try:
            event_id = self._save(camera, args)
        except Exception as exc:
            self.last_recording = {
                "memory_saved": False, "event_id": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[observation] save_failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            self.last_recording = {"memory_saved": True, "event_id": event_id}
            print(f"[observation] saved event={event_id} trigger=live_observation", flush=True)

    def _save(self, camera: dict, args: dict) -> int:
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
        return add_observation(
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
