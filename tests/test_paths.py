import unittest

from video_kb.paths import RunPaths


class RunPathsTests(unittest.TestCase):
    def test_run_id_defaults_to_input_stem(self):
        paths = RunPaths.from_input(r"data\input\demo.mp4")

        self.assertEqual(paths.run_id, "demo")
        self.assertEqual(str(paths.root), str(paths.data_dir / "runs" / "demo"))

    def test_explicit_run_id_still_overrides_default(self):
        paths = RunPaths.from_input(r"data\input\demo.mp4", run_id="custom")

        self.assertEqual(paths.run_id, "custom")


if __name__ == "__main__":
    unittest.main()
