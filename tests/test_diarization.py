import unittest

from video_kb.diarization import clean_diarization_payload


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


if __name__ == "__main__":
    unittest.main()
