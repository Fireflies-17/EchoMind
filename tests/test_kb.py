import unittest

from video_kb.kb import hash_embedding


class KbTests(unittest.TestCase):
    def test_hash_embedding_dimension_and_norm(self):
        vector = hash_embedding("预算 方案 测试", dimension=32)
        self.assertEqual(len(vector), 32)
        norm = sum(x * x for x in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0)


if __name__ == "__main__":
    unittest.main()

