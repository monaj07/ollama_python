import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

TARGET = Path(__file__).resolve().parents[1] / "local_speaker.py"
spec = importlib.util.spec_from_file_location("week11_local_speaker", TARGET)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class SpeakerTests(unittest.TestCase):
    def test_bluetooth_default(self):
        self.assertTrue(module.is_bluetooth_default('node.name = "bluez_output.xx"'))
        self.assertTrue(module.is_bluetooth_default('device.api = "bluez5"'))
        self.assertFalse(module.is_bluetooth_default('node.name = "alsa_output.xxx"'))

    def test_sanitize(self):
        self.assertEqual(module.voice_text('**Answer:** `19`'), 'Answer: 19')
        self.assertEqual(module.voice_text('```python\nprint("secret")\n```'), '')

    def test_no_speaker_skips_without_synthesis(self):
        fake = MagicMock(returncode=0, stdout='node.name = "alsa_output.xxx"')
        with patch.object(module.shutil, 'which', return_value='/usr/bin/wpctl'), patch.object(module.subprocess, 'run', return_value=fake) as run:
            self.assertIn('skipped:', module.speak('19'))
            run.assert_called_once()

    def test_module_piper_uses_week09_python_and_voice_directory(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from types import SimpleNamespace

        with TemporaryDirectory() as root:
            voice = Path(root) / "en_US-test.onnx"
            voice.write_bytes(b"fake voice file")
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "--output_file" in cmd:
                    wav = Path(cmd[cmd.index("--output_file") + 1])
                    wav.write_bytes(b"fake wav")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.dict(module.os.environ, {"PI_SPEAKER_MODEL": str(voice)}, clear=True):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    with patch.object(module.shutil, "which", return_value="/usr/bin/pw-play"):
                        self.assertEqual(module.speak("Hello", check_sink=False), "spoken: Bluetooth speaker")

            self.assertEqual(calls[0][:3], [sys.executable, "-m", "piper"])
            self.assertEqual(calls[0][3:7], ["--data-dir", str(voice.parent), "--model", voice.stem])
            self.assertEqual(calls[1][0], "pw-play")

    def test_default_voice_model_used_without_environment_variable(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        with TemporaryDirectory() as root:
            model = Path(root) / "en_US-lessac-medium.onnx"
            model.write_bytes(b"fake voice file")
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "--output_file" in cmd:
                    Path(cmd[cmd.index("--output_file") + 1]).write_bytes(b"fake wav")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.dict(module.os.environ, {}, clear=True):
                with patch.object(module, "DEFAULT_MODEL", model):
                    with patch.object(module.subprocess, "run", side_effect=fake_run):
                        with patch.object(module.shutil, "which", return_value="/usr/bin/pw-play"):
                            self.assertEqual(module.speak("Hello", check_sink=False), "spoken: Bluetooth speaker")
            self.assertEqual(calls[0][:3], [sys.executable, "-m", "piper"])
            self.assertEqual(calls[0][3:7], ["--data-dir", str(model.parent), "--model", model.stem])

    def test_optional_standalone_piper_is_supported(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from types import SimpleNamespace

        with TemporaryDirectory() as root:
            voice = Path(root) / "test.onnx"
            voice.write_bytes(b"fake")
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "--output_file" in cmd:
                    Path(cmd[cmd.index("--output_file") + 1]).write_bytes(b"fake")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.dict(module.os.environ, {"PI_SPEAKER_MODEL": str(voice), "PI_SPEAKER_PIPER": "/opt/piper/piper"}, clear=True):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    with patch.object(module.shutil, "which", return_value="/usr/bin/pw-play"):
                        self.assertEqual(module.speak("Hello", check_sink=False), "spoken: Bluetooth speaker")
            self.assertEqual(calls[0][:3], ["/opt/piper/piper", "--model", str(voice)])


if __name__ == '__main__':
    unittest.main()
