import random
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from candidate_alchemy import (
    confirm_candidate,
    load_candidate_space,
    make_candidates,
    serialize_candidate_set,
    serialize_state as serialize_craft_state,
    start_state,
)
from semantic_alchemy import (
    DEFAULT_STRENGTH,
    DEFAULT_TARGET,
    MOCK_VECTORS,
    START_WORD,
    TURN_LIMIT,
    load_space,
    step,
    temperature_label,
)


Operation = Literal["mix", "slerp", "subtract", "repel", "purify"]


class CreateGameRequest(BaseModel):
    target: str = Field(DEFAULT_TARGET, examples=[DEFAULT_TARGET])
    seed: int = Field(7, ge=0)
    use_mock: bool = Field(False, description="実モデルを使わず、内蔵モック語彙で開始します。")


class StepRequest(BaseModel):
    operation: Operation = Field("slerp", examples=["slerp"])
    ingredient: str = Field(..., examples=["金"])
    strength: float = Field(DEFAULT_STRENGTH, ge=0.01, le=1.0, examples=[DEFAULT_STRENGTH])


class GameStateResponse(BaseModel):
    game_id: str
    vector_source: str
    current: str
    target: str
    turn: int
    turn_limit: int
    history: list[str]
    similarity: float
    best_similarity: float
    temperature: str
    combo: int
    score_total: float
    recent_events: list[str]
    turns: list[dict]
    complete: bool


class GameResponse(BaseModel):
    state: GameStateResponse
    result: dict | None = None


class CraftCreateGameRequest(BaseModel):
    target: str = Field(DEFAULT_TARGET, examples=[DEFAULT_TARGET])
    difficulty: Literal["normal", "easy"] = Field("normal")
    use_mock: bool = Field(False, description="実モデルを使わず、内蔵モック語彙で開始します。")
    combo_enabled: bool = Field(True)
    goal_bias_enabled: bool = Field(True, description="False の場合は beta=0 の基準挙動になります。")
    seed: int = Field(7, ge=0)


class CraftCandidatesRequest(BaseModel):
    material_a: str = Field(..., examples=[START_WORD])
    material_b: str = Field(..., examples=["金"])
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="素材Aの割合。素材Bは 1-alpha です。")


class CraftConfirmRequest(BaseModel):
    candidate_set_id: str
    candidate_id: str


app = FastAPI(
    title="Semantic Alchemy API",
    description="単語ベクトル錬金ゲームのAPIです。Swagger UIからゲーム開始と1手実行を試せます。",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

games: dict[str, dict] = {}
craft_games: dict[str, dict] = {}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


@app.get("/api/operations")
def operations():
    return {
        "operations": [
            {"id": "mix", "label": "Mix", "description": "安定した重み付き合成"},
            {"id": "slerp", "label": "Slerp", "description": "球面線形補間でなめらかに移動"},
            {"id": "subtract", "label": "Subtract", "description": "素材語の方向を引く"},
            {"id": "repel", "label": "Repel", "description": "目標へ寄せつつ素材語から反発"},
            {"id": "purify", "label": "Purify", "description": "新しい意味軸を足す"},
        ]
    }


@app.get("/api/vocabulary")
def vocabulary(use_mock: bool = True, limit: int = 100):
    if use_mock:
        words = list(MOCK_VECTORS)[:limit]
        return {"vector_source": "mock", "words": words}

    space, source = load_space(use_mock=False)
    return {"vector_source": source, "words": space.words[:limit]}


@app.post("/api/games", response_model=GameResponse)
def create_game(request: CreateGameRequest):
    space, source = load_space(use_mock=request.use_mock)
    target = request.target
    if target not in space:
        raise HTTPException(status_code=400, detail=f"目標語 '{target}' は語彙にありません。")

    game_id = uuid4().hex
    games[game_id] = {
        "space": space,
        "source": source,
        "rng": random.Random(request.seed),
        "state": {
            "current": START_WORD,
            "target": target,
            "history": {START_WORD},
            "turn": 1,
            "combo": 0,
            "score_total": 0.0,
            "best_similarity": 0.0,
            "events": [],
            "turns": [],
        },
    }
    return {"state": serialize_state(game_id), "result": None}


@app.post("/api/craft/games")
def create_craft_game(request: CraftCreateGameRequest):
    try:
        space, source = load_candidate_space(use_mock=request.use_mock)
        session = start_state(
            space,
            source,
            request.target,
            difficulty=request.difficulty,
            combo_enabled=request.combo_enabled,
            goal_bias_enabled=request.goal_bias_enabled,
            seed=request.seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="gensim がインストールされていません。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    game_id = session["game_id"]
    session["state"]["game_id"] = game_id
    craft_games[game_id] = session
    return {"state": serialize_craft_state(session)}


@app.get("/api/craft/games/{game_id}")
def get_craft_game(game_id: str):
    session = get_craft_session(game_id)
    return {"state": serialize_craft_state(session)}


@app.post("/api/craft/games/{game_id}/candidates")
def create_craft_candidates(game_id: str, request: CraftCandidatesRequest):
    session = get_craft_session(game_id)
    result = make_candidates(session, request.material_a, request.material_b, request.alpha)
    if isinstance(result, dict) and result.get("error"):
        status = 422 if result["error"] in {"unsupported_word", "zero_vector", "candidate_shortage"} else 400
        raise HTTPException(status_code=status, detail=result)
    return {"state": serialize_craft_state(session), "candidate_set": serialize_candidate_set(result)}


@app.post("/api/craft/games/{game_id}/confirm")
def confirm_craft_candidate(game_id: str, request: CraftConfirmRequest):
    session = get_craft_session(game_id)
    result = confirm_candidate(session, request.candidate_set_id, request.candidate_id)
    if isinstance(result, dict) and result.get("error"):
        status = 409 if result["error"] in {"stale_candidate_set", "already_confirmed"} else 404
        raise HTTPException(status_code=status, detail=result)
    return {"state": serialize_craft_state(session), "result": result}


@app.get("/api/games/{game_id}", response_model=GameResponse)
def get_game(game_id: str):
    get_session(game_id)
    return {"state": serialize_state(game_id), "result": None}


@app.post("/api/games/{game_id}/steps", response_model=GameResponse)
def create_step(game_id: str, request: StepRequest):
    session = get_session(game_id)
    if session["state"]["turn"] > TURN_LIMIT:
        raise HTTPException(status_code=409, detail="ターン上限に到達しています。")

    result = step(
        session["space"],
        session["state"],
        request.operation,
        request.ingredient,
        session["rng"],
        request.strength,
    )
    return {"state": serialize_state(game_id), "result": serialize_result(result)}


def get_session(game_id: str):
    session = games.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません。")
    return session


def get_craft_session(game_id: str):
    session = craft_games.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません。")
    return session


def serialize_state(game_id: str):
    session = get_session(game_id)
    state = session["state"]
    similarity = session["space"].similarity(state["current"], state["target"])
    best_similarity = max(state.get("best_similarity", similarity), similarity)
    return {
        "game_id": game_id,
        "vector_source": session["source"],
        "current": state["current"],
        "target": state["target"],
        "turn": state["turn"],
        "turn_limit": TURN_LIMIT,
        "history": sorted(state["history"]),
        "similarity": similarity,
        "best_similarity": best_similarity,
        "temperature": temperature_label(similarity),
        "combo": state.get("combo", 0),
        "score_total": state.get("score_total", 0.0),
        "recent_events": state.get("events", []),
        "turns": state.get("turns", []),
        "complete": similarity >= 0.88 or state["turn"] > TURN_LIMIT,
    }


def serialize_result(result):
    data = asdict(result)
    data["candidates"] = [{"word": word, "score": score} for word, score in result.candidates]
    return data
