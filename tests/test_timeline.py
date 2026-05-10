import unittest

from video_kb.timeline import merge_adjacent_segments, overlap_ms


class TimelineTests(unittest.TestCase):
    def test_overlap_ms(self):
        self.assertEqual(overlap_ms(0, 1000, 500, 1500), 500)
        self.assertEqual(overlap_ms(0, 1000, 1000, 2000), 0)

    def test_merge_adjacent_segments(self):
        merged = merge_adjacent_segments(
            [
                {"start_ms": 0, "end_ms": 1000, "speaker": "A", "text": "你好", "source_asr_ids": [1]},
                {"start_ms": 1100, "end_ms": 2000, "speaker": "A", "text": "世界", "source_asr_ids": [2]},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "你好 世界")


if __name__ == "__main__":
    unittest.main()

