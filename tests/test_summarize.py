import unittest

from video_kb.summarize import heuristic_summary


class SummaryTests(unittest.TestCase):
    def test_heuristic_summary_extracts_action_items(self):
        summary = heuristic_summary(
            [
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "speaker": "SPEAKER_00",
                    "text": "我们需要下周完成预算确认。",
                }
            ]
        )
        self.assertEqual(len(summary["knowledge_points"]), 1)
        self.assertEqual(len(summary["action_items"]), 1)


if __name__ == "__main__":
    unittest.main()

