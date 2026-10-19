
"""Event-driven PIR + camera monitoring with persistent memory."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gpiozero import MotionSensor

from memory_store import add_observation, get_active_away_session

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/week_09"))

from tools import PhysicalTools

PIR_GPIO = 23
PIR_SETTLE_S = 60
POLL_INTERVAL_S = 0.2
COOLDOWN_S = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )


def record_motion(
    tools: PhysicalTools,
    session_id: int,
    trigger_time: str,
) -> int:
    """Enrich a PIR event with a camera observation, then save it."""

    try:
        result = tools.execute(
            "observe_scene",
            {"targets": ["person"], "conf": 0.2},
        )
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    counts = result.get("counts")
    count = counts.get("person") if isinstance(counts, dict) else None

    camera_ok = (
        result.get("ok") is True
        and type(count) is int
        and count >= 0
        and isinstance(result.get("detections"), list)
    )

    person_detected = None
    detection_score = None
    evidence_path = None

    details = {
        "pir_signal_active": True,
        "camera_ok": camera_ok,
        "camera_observed_at_utc": result.get("observed_at"),
        "vision_elapsed_s": result.get("tool_elapsed_s"),
    }

    if camera_ok:
        person_detected = count > 0
        scores = [
            float(d["confidence"])
            for d in result["detections"]
            if d.get("label") == "person"
        ]
        detection_score = max(scores, default=None)
        evidence_path = result.get("source")

        details.update({
            "person_count": count,
            "confidence_threshold": result.get(
                "person_count_threshold"
            ),
            "detections": result["detections"],
        })
    else:
        details["vision_error"] = result.get(
            "error", "Invalid camera observation"
        )

    event_id = add_observation(
        trigger="pir_motion",
        away_session_id=session_id,
        observed_at_utc=trigger_time,
        person_detected=person_detected,
        detection_score=detection_score,
        evidence_path=evidence_path,
        details=details,
    )

    print(
        f"[saved] event={event_id} session={session_id} "
        f"person_detected={person_detected} "
        f"score={detection_score}",
        flush=True,
    )

    if not camera_ok:
        print(f"[camera error] {details['vision_error']}", flush=True)

    return event_id


def main() -> None:
    pir = MotionSensor(PIR_GPIO)
    tools = PhysicalTools()

    try:
        print(f"Settling PIR for {PIR_SETTLE_S}s...", flush=True)
        time.sleep(PIR_SETTLE_S)

        previous_active = pir.motion_detected
        last_trigger = float("-inf")

        print("Monitor ready. Press Ctrl+C to stop.", flush=True)

        while True:
            active = pir.motion_detected
            now = time.monotonic()

            # Detect a rising edge, not a continuously HIGH signal.
            if active and not previous_active:
                session = get_active_away_session()

                if session is not None and now - last_trigger >= COOLDOWN_S:
                    last_trigger = now
                    trigger_time = utc_now()

                    print(
                        f"[motion] {trigger_time} "
                        f"session={session['id']}",
                        flush=True,
                    )

                    record_motion(
                        tools=tools,
                        session_id=session["id"],
                        trigger_time=trigger_time,
                    )

            previous_active = active
            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\nMonitor stopped.", flush=True)

    finally:
        tools.close()
        pir.close()


if __name__ == "__main__":
    main()
