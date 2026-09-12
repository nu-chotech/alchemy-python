import unittest

from fastapi import HTTPException

from app import CreateGameRequest, StepRequest, app, create_game, create_step, games


class AppTest(unittest.TestCase):
    def setUp(self):
        games.clear()

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
        self.assertEqual(2, payload["state"]["turn"])

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(HTTPException) as context:
            create_game(CreateGameRequest(target="存在しない語", seed=3, use_mock=True))

        self.assertEqual(400, context.exception.status_code)

    def test_openapi_schema_is_generated(self):
        schema = app.openapi()

        self.assertIn("/api/games", schema["paths"])
        self.assertIn("/api/games/{game_id}/steps", schema["paths"])


if __name__ == "__main__":
    unittest.main()
