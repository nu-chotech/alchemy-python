import unittest

from candidate_alchemy import make_candidates, start_state
from semantic_alchemy import MOCK_VECTORS, START_WORD, VectorSpace


class CandidateAlchemyTest(unittest.TestCase):
    def setUp(self):
        self.space = VectorSpace(MOCK_VECTORS)

    def make_session(self, **kwargs):
        return start_state(self.space, "mock", "100億", **kwargs)

    def test_candidates_do_not_advance_state(self):
        session = self.make_session(difficulty="normal")

        candidate_set = make_candidates(session, START_WORD, "金", 0.5)

        self.assertEqual(1, session["state"]["turn"])
        self.assertEqual([START_WORD], session["state"]["history"])
        self.assertEqual(3, len(candidate_set.candidates))
        self.assertIsNone(candidate_set.candidates[0].target_similarity)

    def test_easy_candidates_include_target_hint(self):
        session = self.make_session(difficulty="easy")

        candidate_set = make_candidates(session, START_WORD, "金", 0.5)

        self.assertIsInstance(candidate_set.candidates[0].target_similarity, float)

    def test_goal_bias_can_be_disabled_for_beta_zero(self):
        session = self.make_session(difficulty="easy", goal_bias_enabled=False)

        candidate_set = make_candidates(session, START_WORD, "金", 0.5)

        self.assertEqual(0.0, candidate_set.beta)

    def test_unknown_material_returns_suggestions(self):
        session = self.make_session(difficulty="normal")

        result = make_candidates(session, START_WORD, "存在しない語", 0.5)

        self.assertEqual("unsupported_word", result["error"])
        self.assertIn(START_WORD, result["suggestions"])


if __name__ == "__main__":
    unittest.main()
