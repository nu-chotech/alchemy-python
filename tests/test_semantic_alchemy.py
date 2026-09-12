import random
import unittest

from semantic_alchemy import MOCK_VECTORS, START_WORD, VectorSpace, step


class SemanticAlchemyTest(unittest.TestCase):
    def setUp(self):
        self.space = VectorSpace(MOCK_VECTORS)

    def test_step_is_reproducible_with_seed(self):
        left = {"current": START_WORD, "target": "100億", "history": {START_WORD}, "turn": 1}
        right = {"current": START_WORD, "target": "100億", "history": {START_WORD}, "turn": 1}

        left_result = step(self.space, left, "slerp", "金", random.Random(10))
        right_result = step(self.space, right, "slerp", "金", random.Random(10))

        self.assertTrue(left_result.accepted)
        self.assertEqual(left_result.result_word, right_result.result_word)
        self.assertEqual(left_result.breakdown, right_result.breakdown)

    def test_unknown_word_is_rejected_without_advancing_turn(self):
        state = {"current": START_WORD, "target": "100億", "history": {START_WORD}, "turn": 1}

        result = step(self.space, state, "mix", "存在しない語", random.Random(1))

        self.assertFalse(result.accepted)
        self.assertEqual("語彙にありません", result.message)
        self.assertEqual(1, state["turn"])
        self.assertEqual(START_WORD, state["current"])

    def test_unknown_operation_is_rejected(self):
        state = {"current": START_WORD, "target": "100億", "history": {START_WORD}, "turn": 1}

        result = step(self.space, state, "warp", "金", random.Random(1))

        self.assertFalse(result.accepted)
        self.assertEqual("未知の演算です", result.message)

    def test_successful_step_returns_score_breakdown(self):
        state = {"current": START_WORD, "target": "100億", "history": {START_WORD}, "turn": 1}

        result = step(self.space, state, "repel", "借金", random.Random(2), strength=0.62)

        self.assertTrue(result.accepted)
        self.assertEqual(0.62, result.strength)
        self.assertGreaterEqual(result.multiplier, 1.0)
        self.assertIsInstance(result.awards, list)
        self.assertIn("score_total", state)
        self.assertGreater(result.score, 0)
        self.assertEqual({"target", "coherence", "rarity", "novelty", "risk"}, set(result.breakdown))
        self.assertIn(result.result_word, state["history"])
        self.assertEqual(2, state["turn"])


if __name__ == "__main__":
    unittest.main()
