import unittest

from fastapi import HTTPException

from app import (
    CraftCandidatesRequest,
    CraftConfirmRequest,
    CraftCreateGameRequest,
    CreateGameRequest,
    StepRequest,
    app,
    craft_games,
    create_craft_candidates,
    create_craft_game,
    create_game,
    create_step,
    confirm_craft_candidate,
    games,
)


class AppTest(unittest.TestCase):
    def setUp(self):
        games.clear()
        craft_games.clear()

    def test_create_mock_game(self):
        payload = create_game(CreateGameRequest(target="100億", seed=3, use_mock=True))

        self.assertEqual("りんご", payload["state"]["current"])
        self.assertEqual("100億", payload["state"]["target"])
        self.assertIsNone(payload["result"])

    def test_step_mock_game(self):
        created = create_game(CreateGameRequest(target="100億", seed=3, use_mock=True))
        game_id = created["state"]["game_id"]

        payload = create_step(game_id, StepRequest(operation="slerp", ingredient="金", strength=0.44))

        self.assertTrue(payload["result"]["accepted"])
        self.assertEqual(0.44, payload["result"]["strength"])
        self.assertIn("multiplier", payload["result"])
        self.assertIn("recent_events", payload["state"])
        self.assertEqual("りんご", payload["state"]["turns"][0]["before_word"])
        self.assertEqual("金", payload["state"]["turns"][0]["ingredient"])
        self.assertEqual(payload["state"]["current"], payload["state"]["turns"][0]["after_word"])
        self.assertEqual(2, payload["state"]["turn"])

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(HTTPException) as context:
            create_game(CreateGameRequest(target="存在しない語", seed=3, use_mock=True))

        self.assertEqual(400, context.exception.status_code)

    def test_openapi_schema_is_generated(self):
        schema = app.openapi()

        self.assertIn("/api/games", schema["paths"])
        self.assertIn("/api/games/{game_id}/steps", schema["paths"])
        self.assertIn("/api/craft/games", schema["paths"])

    def test_craft_candidate_confirm_advances_once(self):
        created = create_craft_game(CraftCreateGameRequest(target="100億", use_mock=True))
        game_id = created["state"]["game_id"]
        candidates = create_craft_candidates(
            game_id,
            CraftCandidatesRequest(material_a="りんご", material_b="金", alpha=0.5),
        )

        self.assertEqual(1, candidates["state"]["turn"])
        self.assertEqual(["りんご"], candidates["state"]["history"])

        candidate_set_id = candidates["candidate_set"]["candidate_set_id"]
        candidate_id = candidates["candidate_set"]["candidates"][0]["id"]
        confirmed = confirm_craft_candidate(
            game_id,
            CraftConfirmRequest(candidate_set_id=candidate_set_id, candidate_id=candidate_id),
        )

        self.assertEqual(2, confirmed["state"]["turn"])
        self.assertEqual(2, len(confirmed["state"]["history"]))
        self.assertIn("target_similarity", confirmed["result"])

        with self.assertRaises(HTTPException) as context:
            confirm_craft_candidate(
                game_id,
                CraftConfirmRequest(candidate_set_id=candidate_set_id, candidate_id=candidate_id),
            )
        self.assertEqual(409, context.exception.status_code)

    def test_craft_stale_candidate_set_is_rejected(self):
        created = create_craft_game(CraftCreateGameRequest(target="100億", use_mock=True))
        game_id = created["state"]["game_id"]
        first = create_craft_candidates(game_id, CraftCandidatesRequest(material_a="りんご", material_b="金", alpha=0.5))
        create_craft_candidates(game_id, CraftCandidatesRequest(material_a="りんご", material_b="宇宙", alpha=0.5))

        with self.assertRaises(HTTPException) as context:
            confirm_craft_candidate(
                game_id,
                CraftConfirmRequest(
                    candidate_set_id=first["candidate_set"]["candidate_set_id"],
                    candidate_id=first["candidate_set"]["candidates"][0]["id"],
                ),
            )

        self.assertEqual(409, context.exception.status_code)

    def test_craft_easy_mode_exposes_target_similarity_but_normal_hides_it(self):
        normal = create_craft_game(CraftCreateGameRequest(target="100億", difficulty="normal", use_mock=True))
        easy = create_craft_game(CraftCreateGameRequest(target="100億", difficulty="easy", use_mock=True))

        normal_candidates = create_craft_candidates(
            normal["state"]["game_id"],
            CraftCandidatesRequest(material_a="りんご", material_b="金", alpha=0.5),
        )
        easy_candidates = create_craft_candidates(
            easy["state"]["game_id"],
            CraftCandidatesRequest(material_a="りんご", material_b="金", alpha=0.5),
        )

        self.assertIsNone(normal_candidates["candidate_set"]["candidates"][0]["target_similarity"])
        self.assertIsInstance(easy_candidates["candidate_set"]["candidates"][0]["target_similarity"], float)


if __name__ == "__main__":
    unittest.main()
