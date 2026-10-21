"""Unit checks for safe Telegram voice-file handling (no actual Whisper invocation)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telegram_stt import validated_voice_file


class TelegramVoiceFileTests(unittest.TestCase):
    def test_valid_file_in_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voice = root / "1234-AbCd-voice.oga"
            voice.write_bytes(b"fakeaudio")
            self.assertEqual(validated_voice_file(voice, root), voice)

    def test_outside_project_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inside = root / "project"
            inside.mkdir()
            voice = root / "1234-AbCd-voice.oga"
            voice.write_bytes(b"fakeaudio")
            with self.assertRaises(ValueError):
                validated_voice_file(voice, inside)

    def test_non_voice_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "secret.txt"
            file.write_bytes(b"hello")
            with self.assertRaises(ValueError):
                validated_voice_file(file, root)

    def test_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "1234-AbCd-voice.oga"
            target.write_bytes(b"fakeaudio")
            alias = root / "5678-AbCd-voice.oga"
            alias.symlink_to(target)
            with self.assertRaises(ValueError):
                validated_voice_file(alias, root)

    def test_empty_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "1234-AbCd-voice.oga"
            file.write_bytes(b"")
            with self.assertRaises(ValueError):
                validated_voice_file(file, root)


if __name__ == "__main__":
    unittest.main()
