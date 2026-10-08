import time
from datetime import datetime, timezone

from gpiozero import MotionSensor


class MotionTools:
    def __init__(self):
        self.pir = MotionSensor(23)

    def detect_motion(self) -> dict:
        """Read the PIR's current motion output.

        An active signal indicates detected movement within the
        sensor's hold period. An inactive signal does not prove
        that the room is empty.
        """
        active = bool(self.pir.motion_detected)
        return {
            "ok": True,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "motion_signal_active": active,
        }

    def close(self):
        self.pir.close()


if __name__ == "__main__":
    tools = MotionTools()
    try:
        print("Allowing 60 seconds for the PIR to settle...", flush=True)
        time.sleep(60)

        while True:
            input("Press Enter to read motion; Ctrl+C to quit: ")
            print(tools.detect_motion())

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tools.close()