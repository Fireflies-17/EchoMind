import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_kb.diarization import clean_diarization_payload, diarize, _segments_from_3dspeaker_json


class DiarizationCleaningTests(unittest.TestCase):
    def test_cleaning_drops_short_fragments_and_limits_speakers(self):
        payload = {
            "segments": [
                {"start_ms": 0, "end_ms": 1000, "speaker": "SPEAKER_00"},
                {"start_ms": 1100, "end_ms": 2000, "speaker": "SPEAKER_00"},
                {"start_ms": 2050, "end_ms": 2200, "speaker": "SPEAKER_03"},
                {"start_ms": 2400, "end_ms": 3400, "speaker": "SPEAKER_01"},
                {"start_ms": 3500, "end_ms": 4500, "speaker": "SPEAKER_02"},
            ]
        }

        cleaned = clean_diarization_payload(
            payload,
            min_segment_ms=500,
            merge_gap_ms=200,
            max_speakers=2,
            reassign_gap_ms=1000,
        )

        speakers = {segment["speaker"] for segment in cleaned["segments"]}
        self.assertEqual(speakers, {"SPEAKER_00", "SPEAKER_01"})
        self.assertEqual(cleaned["cleaning"]["dropped_short_segments"], 1)
        self.assertEqual(cleaned["cleaning"]["reassigned_segments"], 1)

    def test_skipped_payload_is_preserved(self):
        payload = {
            "skipped": True,
            "segments": [{"start_ms": 0, "end_ms": 1000, "speaker": "SPEAKER_00"}],
        }

        cleaned = clean_diarization_payload(payload)

        self.assertTrue(cleaned["skipped"])
        self.assertEqual(cleaned["segments"][0]["speaker"], "SPEAKER_00")
        self.assertTrue(cleaned["cleaning"]["skipped"])

    def test_3dspeaker_backend_normalizes_output(self):
        class Fake3DSpeaker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, audio_path, speaker_num=None):
                return [[0.1, 1.2, 0], [1.3, 2.5, 1]]

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "speakers.json"
            with patch("video_kb.diarization._import_3dspeaker", return_value=Fake3DSpeaker):
                payload = diarize(
                    "audio.wav",
                    output,
                    backend="3dspeaker",
                    speaker_num=2,
                    device="cpu",
                )

        self.assertEqual(payload["backend"], "3dspeaker")
        self.assertEqual(payload["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(payload["segments"][1]["speaker"], "SPEAKER_01")

    def test_3dspeaker_json_output_is_normalized(self):
        payload = {
            "audio_0.1_1.2": {"start": 0.1, "stop": 1.2, "speaker": 0},
            "audio_1.3_2.5": {"start": 1.3, "stop": 2.5, "speaker": 1},
        }

        segments = _segments_from_3dspeaker_json(payload)

        self.assertEqual(segments[0], {"start_ms": 100, "end_ms": 1200, "speaker": "SPEAKER_00"})
        self.assertEqual(segments[1], {"start_ms": 1300, "end_ms": 2500, "speaker": "SPEAKER_01"})


if __name__ == "__main__":
    unittest.main()
