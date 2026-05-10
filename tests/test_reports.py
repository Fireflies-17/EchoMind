import unittest

from video_kb.reports import build_speaker_report, build_topics_report, group_by_speaker_turn


class ReportTests(unittest.TestCase):
    def test_group_by_speaker_turn_merges_until_speaker_changes(self):
        timeline = [
            {"source_index": 0, "start_ms": 0, "end_ms": 1000, "speaker": "SPEAKER_00", "text": "第一句"},
            {"source_index": 1, "start_ms": 1000, "end_ms": 2000, "speaker": "SPEAKER_00", "text": "第二句"},
            {"source_index": 2, "start_ms": 2100, "end_ms": 3000, "speaker": "SPEAKER_01", "text": "回应"},
        ]

        turns = group_by_speaker_turn(timeline)

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(turns[0]["text"], "第一句 第二句")

    def test_speaker_report_accepts_numeric_speaker_suffix(self):
        report = build_speaker_report("tests/fixtures/sample_timeline.json", speaker="1", engine="heuristic")

        self.assertEqual(report["speaker"], "SPEAKER_01")
        self.assertEqual(len(report["segments"]), 1)
        self.assertEqual(report["segments"][0]["speaker"], "SPEAKER_01")

    def test_topics_report_has_speaker_blocks(self):
        report = build_topics_report("tests/fixtures/sample_timeline.json", engine="heuristic")

        self.assertEqual(report["mode"], "topics")
        self.assertGreaterEqual(len(report["topics"]), 1)
        self.assertIn("speakers", report["topics"][0])


if __name__ == "__main__":
    unittest.main()
