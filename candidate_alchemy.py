import math
import random
from dataclasses import dataclass
from hashlib import sha1
from uuid import uuid4

import numpy as np

from semantic_alchemy import (
    DEFAULT_TARGET,
    MOCK_VECTORS,
    MODEL_PATH,
    START_WORD,
    GensimSpace,
    VectorSpace,
    normalize,
    temperature_label,
)


POOL_SIZE = 24
CANDIDATE_COUNT = 3
NORMAL_BETA = 0.03
EASY_BETA = 0.12
COMBO_BETA_STEP = 0.025
MAX_BETA = 0.22
MAP_ANCHOR_LIMIT = 120
SUGGESTION_LIMIT = 8


@dataclass(frozen=True)
class CandidateOption:
    id: str
    word: str
    rank: int
    blend_similarity: float
    target_similarity: float | None
    score: float
    point: dict[str, float]


@dataclass(frozen=True)
class CandidateSet:
    id: str
    material_a: str
    material_b: str
    alpha: float
    beta: float
    pool_size: int
    candidates: list[CandidateOption]
    status: str = "open"


class Projection3D:
    def __init__(self, space, seed_words):
        self.space = space
        self.origin, self.components, self.projection_id = build_projection(space, seed_words)

    def point(self, word):
        vector = normalize(self.space[word])
        centered = vector - self.origin
        coords = centered @ self.components.T
        return {
            "x": float(coords[0]),
            "y": float(coords[1]),
            "z": float(coords[2]),
        }


def load_candidate_space(use_mock=False):
    if use_mock:
        return VectorSpace(MOCK_VECTORS), "mock"

    from gensim.models import KeyedVectors

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} が見つかりません。モックモードを明示して起動してください。")

    return GensimSpace(KeyedVectors.load(str(MODEL_PATH))), str(MODEL_PATH)


def start_state(space, source, target, difficulty="normal", combo_enabled=True, goal_bias_enabled=True, seed=7):
    if target not in space:
        raise ValueError(f"目標語 '{target}' は語彙にありません。")
    if START_WORD not in space:
        raise ValueError(f"開始語 '{START_WORD}' は語彙にありません。")
    if difficulty not in {"normal", "easy"}:
        raise ValueError("難易度は normal または easy を指定してください。")

    projection = Projection3D(space, map_seed_words(space, target))
    return {
        "game_id": uuid4().hex,
        "space": space,
        "source": source,
        "projection": projection,
        "rng": random.Random(seed),
        "state": {
            "current": START_WORD,
            "target": target,
            "difficulty": difficulty,
            "combo_enabled": bool(combo_enabled),
            "goal_bias_enabled": bool(goal_bias_enabled),
            "turn": 1,
            "combo": 0,
            "last_confirmed_similarity": None,
            "history": [START_WORD],
            "candidate_sets": {},
            "active_candidate_set_id": None,
            "events": [],
            "complete": START_WORD == target,
        },
    }


def make_candidates(session, material_a, material_b, alpha):
    space = session["space"]
    state = session["state"]
    material_a = material_a.strip()
    material_b = material_b.strip()
    alpha = clamp(float(alpha), 0.0, 1.0)

    missing = [word for word in (material_a, material_b) if word not in space]
    if missing:
        return {
            "error": "unsupported_word",
            "message": f"語彙にない素材があります: {', '.join(missing)}",
            "suggestions": vocabulary_suggestions(space),
        }

    blended = alpha * normalize(space[material_a]) + (1.0 - alpha) * normalize(space[material_b])
    norm = float(np.linalg.norm(blended))
    if norm == 0.0:
        return {
            "error": "zero_vector",
            "message": "素材ベクトルが打ち消し合い、合成位置を作れません。",
            "suggestions": vocabulary_suggestions(space),
        }
    query = blended / norm
    beta = beta_for_state(state)
    excluded = {material_a, material_b}
    pool = space.nearest(query, topn=POOL_SIZE, exclude=excluded)

    if not pool:
        return {
            "error": "candidate_shortage",
            "message": "候補語を作れませんでした。別の素材か割合を試してください。",
            "suggestions": vocabulary_suggestions(space),
        }

    target = state["target"]
    scored = []
    for word, blend_similarity in pool:
        target_similarity = space.similarity(word, target)
        score = (1.0 - beta) * blend_similarity + beta * target_similarity
        scored.append((word, float(blend_similarity), float(target_similarity), float(score)))
    scored.sort(key=lambda item: (-item[3], -item[1], item[0]))

    candidate_set_id = uuid4().hex
    candidates = [
        CandidateOption(
            id=stable_candidate_id(candidate_set_id, word),
            word=word,
            rank=index + 1,
            blend_similarity=blend_similarity,
            target_similarity=target_similarity if state["difficulty"] == "easy" else None,
            score=score,
            point=session["projection"].point(word),
        )
        for index, (word, blend_similarity, target_similarity, score) in enumerate(scored[:CANDIDATE_COUNT])
    ]
    if not candidates:
        return {
            "error": "candidate_shortage",
            "message": "候補語を作れませんでした。別の素材か割合を試してください。",
            "suggestions": vocabulary_suggestions(space),
        }

    candidate_set = CandidateSet(candidate_set_id, material_a, material_b, alpha, beta, len(pool), candidates)
    state["candidate_sets"][candidate_set_id] = candidate_set
    state["active_candidate_set_id"] = candidate_set_id
    return candidate_set


def confirm_candidate(session, candidate_set_id, candidate_id):
    state = session["state"]
    candidate_set = state["candidate_sets"].get(candidate_set_id)
    if candidate_set is None:
        return {"error": "unknown_candidate_set", "message": "候補セットが見つかりません。"}
    if state.get("active_candidate_set_id") != candidate_set_id:
        return {"error": "stale_candidate_set", "message": "古い候補セットは確定できません。"}
    if candidate_set.status != "open":
        return {"error": "already_confirmed", "message": "この候補セットはすでに確定済みです。"}

    selected = next((candidate for candidate in candidate_set.candidates if candidate.id == candidate_id), None)
    if selected is None:
        return {"error": "unknown_candidate", "message": "候補セット内に選択語がありません。"}

    space = session["space"]
    target = state["target"]
    previous_similarity = state["last_confirmed_similarity"]
    current_similarity = space.similarity(selected.word, target)
    combo_delta = None if previous_similarity is None else current_similarity - previous_similarity
    if previous_similarity is not None and state["combo_enabled"]:
        state["combo"] = state["combo"] + 1 if combo_delta > 0.0 else 0
    elif previous_similarity is not None:
        state["combo"] = 0

    state["current"] = selected.word
    state["history"].append(selected.word)
    state["turn"] += 1
    state["last_confirmed_similarity"] = current_similarity
    state["active_candidate_set_id"] = None
    state["complete"] = selected.word == target
    state["events"].append(
        {
            "turn": state["turn"] - 1,
            "word": selected.word,
            "combo": state["combo"],
            "target_similarity": current_similarity,
        }
    )
    state["candidate_sets"][candidate_set_id] = CandidateSet(
        candidate_set.id,
        candidate_set.material_a,
        candidate_set.material_b,
        candidate_set.alpha,
        candidate_set.beta,
        candidate_set.pool_size,
        candidate_set.candidates,
        "confirmed",
    )

    return {
        "word": selected.word,
        "point": selected.point,
        "target_similarity": current_similarity,
        "combo_delta": combo_delta,
        "combo": state["combo"],
        "turn": state["turn"],
        "complete": state["complete"],
    }


def serialize_state(session):
    state = session["state"]
    space = session["space"]
    current_similarity = space.similarity(state["current"], state["target"])
    return {
        "game_id": state.get("game_id", session["game_id"]),
        "vector_source": session["source"],
        "coordinate_system": session["projection"].projection_id,
        "current": state["current"],
        "target": state["target"],
        "difficulty": state["difficulty"],
        "combo_enabled": state["combo_enabled"],
        "goal_bias_enabled": state["goal_bias_enabled"],
        "turn": state["turn"],
        "history": state["history"],
        "current_similarity": current_similarity,
        "confirmed_similarity": state["last_confirmed_similarity"],
        "temperature": temperature_label(current_similarity),
        "combo": state["combo"],
        "complete": state["complete"],
        "nodes": [
            {"word": state["target"], "role": "target", "point": session["projection"].point(state["target"])},
            {"word": state["current"], "role": "current", "point": session["projection"].point(state["current"])},
        ],
        "events": state["events"][-8:],
    }


def serialize_candidate_set(candidate_set):
    return {
        "candidate_set_id": candidate_set.id,
        "material_a": candidate_set.material_a,
        "material_b": candidate_set.material_b,
        "alpha": candidate_set.alpha,
        "beta": candidate_set.beta,
        "pool_size": candidate_set.pool_size,
        "candidates": [
            {
                "id": candidate.id,
                "word": candidate.word,
                "rank": candidate.rank,
                "blend_similarity": candidate.blend_similarity,
                "target_similarity": candidate.target_similarity,
                "score": candidate.score,
                "point": candidate.point,
            }
            for candidate in candidate_set.candidates
        ],
    }


def beta_for_state(state):
    if not state["goal_bias_enabled"]:
        return 0.0
    base = EASY_BETA if state["difficulty"] == "easy" else NORMAL_BETA
    combo_bonus = COMBO_BETA_STEP * state["combo"] if state["combo_enabled"] else 0.0
    return min(MAX_BETA, base + combo_bonus)


def map_seed_words(space, target):
    words = [START_WORD, target]
    if hasattr(space, "words"):
        for word in space.words:
            if word not in words:
                words.append(word)
            if len(words) >= MAP_ANCHOR_LIMIT:
                break
    return words


def build_projection(space, seed_words):
    vectors = np.array([normalize(space[word]) for word in seed_words if word in space], dtype=float)
    if len(vectors) < 3:
        vectors = np.vstack([vectors, np.eye(vectors.shape[1])[: 3 - len(vectors)]])
    origin = vectors.mean(axis=0)
    centered = vectors - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]
    if components.shape[0] < 3:
        padding = np.eye(vectors.shape[1])[: 3 - components.shape[0]]
        components = np.vstack([components, padding])
    projection_hash = sha1("|".join(seed_words[:MAP_ANCHOR_LIMIT]).encode("utf-8")).hexdigest()[:10]
    return origin, components, f"pca-{projection_hash}"


def vocabulary_suggestions(space):
    if hasattr(space, "words"):
        return space.words[:SUGGESTION_LIMIT]
    return list(MOCK_VECTORS)[:SUGGESTION_LIMIT]


def stable_candidate_id(candidate_set_id, word):
    return sha1(f"{candidate_set_id}:{word}".encode("utf-8")).hexdigest()[:16]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))
