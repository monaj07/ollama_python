"""Verify both LEDs and the button before introducing a model. Run on the Pi."""
from time import sleep


def main():
    from gpiozero import Button, LED

    with (
        LED(17, initial_value=False) as green,
        LED(27, initial_value=False) as red,
        Button(22, pull_up=True, bounce_time=0.05) as button,
    ):
        try:
            print("Green LED test")
            green.on()
            sleep(2)
            green.off()
            print("Red LED test")
            red.on()
            sleep(2)
            red.off()
            print("Press and release the button; Ctrl+C to exit.")
            while True:
                button.wait_for_press()
                print("Button pressed")
                green.on()
                button.wait_for_release()
                green.off()
                print("Button released")
        except KeyboardInterrupt:
            pass
        finally:
            green.off()
            red.off()


if __name__ == "__main__":
    main()
