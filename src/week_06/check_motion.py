import time

from gpiozero import MotionSensor

# BCM GPIO23 = physical Pi pin 16.
pir = MotionSensor(23)

try:
    print("Allowing 60 seconds for the PIR to settle...", flush=True)
    time.sleep(60)
    print("Ready. Walk across the sensor's field of view.", flush=True)

    while True:
        timestamp = time.strftime("%H:%M:%S")
        print(
            f"{timestamp} | motion_signal_active={pir.motion_detected}",
            flush=True,
        )
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    pir.close()