import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_PATH = Path("model/entity_vector.kv")
START_WORD = "りんご"
DEFAULT_TARGET = "100億"
TURN_LIMIT = 8
TOPN = 12
TEMPERATURE = 0.18
DEFAULT_STRENGTH = 0.44

TARGET_WEIGHT = 6800
COHERENCE_WEIGHT = 1300
RARITY_WEIGHT = 1900
NOVELTY_WEIGHT = 900
RISK_PENALTY = 1700

MOCK_VECTORS = {
    "りんご": [0.8, 0.18, 0.1, 0.1, 0.0],
    "果実": [0.82, 0.16, 0.08, 0.08, 0.0],
    "金": [0.1, 0.85, 0.18, 0.05, 0.1],
    "市場": [0.12, 0.75, 0.2, 0.08, 0.24],
    "投資": [0.04, 0.88, 0.18, 0.05, 0.28],
    "宇宙": [0.06, 0.2, 0.88, 0.25, 0.18],
    "ロケット": [0.1, 0.22, 0.82, 0.2, 0.15],
    "夢": [0.2, 0.22, 0.32, 0.84, 0.25],
    "発明": [0.18, 0.42, 0.34, 0.62, 0.42],
    "王国": [0.25, 0.6, 0.28, 0.58, 0.18],
    "税金": [0.02, 0.78, 0.12, 0.02, 0.12],
    "借金": [0.0, 0.7, 0.08, -0.12, 0.42],
    "秘密": [0.14, 0.28, 0.24, 0.74, 0.72],
    "陰謀": [0.02, 0.32, 0.18, 0.62, 0.84],
    "伝説": [0.3, 0.4, 0.42, 0.76, 0.32],
    "錬金術": [0.38, 0.56, 0.36, 0.72, 0.48],
    "会社": [0.08, 0.78, 0.12, 0.12, 0.22],
    "国家": [0.04, 0.82, 0.2, 0.24, 0.22],
    "革命": [0.0, 0.52, 0.3, 0.74, 0.56],
    "100億": [0.04, 0.94, 0.28, 0.34, 0.34],
}


@dataclass(frozen=True)
class StepResult:
    turn: int
    operation: str
    ingredient: str
    strength: float
    result_word: str | None
    accepted: bool
    message: str
    score: float
    breakdown: dict[str, float]
    candidates: list[tuple[str, float]]


class VectorSpace:
    def __init__(self, vectors):
        self.vectors = {word: normalize(np.array(vector, dtype=float)) for word, vector in vectors.items()}

    def __contains__(self, word):
        return word in self.vectors

    def __getitem__(self, word):
        return self.vectors[word]

    @property
    def words(self):
        return list(self.vectors)

    def similarity(self, left, right):
        return cosine(self[left], self[right])

    def nearest(self, vector, topn=TOPN, exclude=()):
        excluded = set(exclude)
        vector = normalize(vector)
        scored = [
            (word, cosine(vector, word_vector))
            for word, word_vector in self.vectors.items()
            if word not in excluded
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topn]


class GensimSpace:
    def __init__(self, keyed_vectors):
        self.model = keyed_vectors

    def __contains__(self, word):
        return word in self.model

    def __getitem__(self, word):
        return self.model[word]

    def similarity(self, left, right):
        return float(self.model.similarity(left, right))

    def nearest(self, vector, topn=TOPN, exclude=()):
        excluded = set(exclude)
        results = []
        for word, score in self.model.similar_by_vector(normalize(vector), topn=topn + len(excluded) + 8):
            if word not in excluded:
                results.append((word, float(score)))
            if len(results) >= topn:
                break
        return results


def load_space(model_path=MODEL_PATH, use_mock=False):
    if use_mock:
        return VectorSpace(MOCK_VECTORS), "mock"

    try:
        from gensim.models import KeyedVectors

        return GensimSpace(KeyedVectors.load(str(model_path))), str(model_path)
    except Exception as exc:
        print(f"実モデルを読み込めませんでした: {exc}")
        print("モックベクトルで実験を続けます。")
        return VectorSpace(MOCK_VECTORS), "mock"


def normalize(vector):
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine(left, right):
    left = normalize(left)
    right = normalize(right)
    return float(np.dot(left, right))


def slerp(left, right, amount):
    left = normalize(left)
    right = normalize(right)
    dot = max(-1.0, min(1.0, float(np.dot(left, right))))
    omega = math.acos(dot)
    if abs(omega) < 1e-6:
        return normalize((1.0 - amount) * left + amount * right)

    sin_omega = math.sin(omega)
    return normalize(
        math.sin((1.0 - amount) * omega) / sin_omega * left
        + math.sin(amount * omega) / sin_omega * right
    )


def orthogonal_component(vector, basis):
    basis = normalize(basis)
    return vector - np.dot(vector, basis) * basis


def operate(current, ingredient, target, operation, strength=DEFAULT_STRENGTH):
    strength = max(0.01, min(1.0, float(strength)))
    if operation == "mix":
        return normalize((1.0 - strength) * current + strength * ingredient)
    if operation == "slerp":
        return slerp(current, ingredient, strength)
    if operation == "subtract":
        return normalize(current - strength * ingredient)
    if operation == "repel":
        return normalize(current + strength * target - (strength * 0.78) * ingredient)
    if operation == "purify":
        return normalize(current + strength * orthogonal_component(ingredient, current))
    raise ValueError(f"unknown operation: {operation}")


def density(space, word, neighbors=6):
    near = space.nearest(space[word], topn=neighbors + 1, exclude={word})
    if not near:
        return 0.0
    return sum(score for _, score in near[:neighbors]) / min(neighbors, len(near))


def score_word(space, word, current_word, ingredient_word, target_word, operation):
    target_sim = clamp01((space.similarity(word, target_word) + 1.0) / 2.0)
    coherence = clamp01(
        ((space.similarity(word, current_word) + 1.0) / 2.0
        + (space.similarity(word, ingredient_word) + 1.0) / 2.0)
        / 2.0
    )
    rarity = clamp01(1.0 - density(space, word))
    novelty = clamp01(1.0 - (space.similarity(word, current_word) + 1.0) / 2.0)
    risk = risk_for(operation, target_sim, coherence)

    score = (
        TARGET_WEIGHT * target_sim
        + COHERENCE_WEIGHT * coherence
        + RARITY_WEIGHT * rarity
        + NOVELTY_WEIGHT * novelty
        - RISK_PENALTY * risk
    )
    score = max(0.0, score)
    return score, {
        "target": target_sim,
        "coherence": coherence,
        "rarity": rarity,
        "novelty": novelty,
        "risk": risk,
    }


def risk_for(operation, target_sim, coherence):
    base = {
        "mix": 0.05,
        "slerp": 0.08,
        "purify": 0.16,
        "repel": 0.28,
        "subtract": 0.34,
    }[operation]
    instability = max(0.0, 0.55 - coherence) * 0.9
    greed = max(0.0, target_sim - 0.82) ** 2.0
    return clamp01(base + instability + greed)


def choose_candidate(candidates, rng):
    if not candidates:
        return None
    weights = [math.exp(score / TEMPERATURE) for _, score in candidates]
    total = sum(weights)
    pick = rng.random() * total
    upto = 0.0
    for item, weight in zip(candidates, weights):
        upto += weight
        if upto >= pick:
            return item
    return candidates[-1]


def step(space, state, operation, ingredient, rng, strength=DEFAULT_STRENGTH):
    current = state["current"]
    target = state["target"]
    history = state["history"]
    strength = max(0.01, min(1.0, float(strength)))

    if ingredient not in space:
        return StepResult(state["turn"], operation, ingredient, strength, None, False, "語彙にありません", 0.0, {}, [])
    if operation not in {"mix", "slerp", "subtract", "repel", "purify"}:
        return StepResult(state["turn"], operation, ingredient, strength, None, False, "未知の演算です", 0.0, {}, [])

    vector = operate(space[current], space[ingredient], space[target], operation, strength)
    candidates = space.nearest(vector, topn=TOPN, exclude=history | {current, ingredient})
    chosen = choose_candidate(candidates, rng)
    if chosen is None:
        return StepResult(state["turn"], operation, ingredient, strength, None, False, "候補がありません", 0.0, {}, [])

    result_word, _ = chosen
    score, breakdown = score_word(space, result_word, current, ingredient, target, operation)
    state["current"] = result_word
    state["history"].add(result_word)
    state["turn"] += 1
    return StepResult(state["turn"] - 1, operation, ingredient, strength, result_word, True, "錬成成功", score, breakdown, candidates[:5])


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def print_status(space, state):
    current = state["current"]
    target = state["target"]
    sim = space.similarity(current, target)
    print(f"\nTurn {state['turn']} / {TURN_LIMIT}")
    print(f"現在語: {current}  目標概念: {target}  温度: {temperature_label(sim)} ({sim:.3f})")
    print("演算: mix / slerp / subtract / repel / purify")


def print_result(result):
    if not result.accepted:
        print(f"失敗: {result.message}")
        return

    print(f"=> {result.result_word}  落札価格: {result.score:,.0f}")
    print(
        "内訳: "
        f"目標 {result.breakdown['target']:.3f}, "
        f"整合 {result.breakdown['coherence']:.3f}, "
        f"希少 {result.breakdown['rarity']:.3f}, "
        f"意外 {result.breakdown['novelty']:.3f}, "
        f"危険 {result.breakdown['risk']:.3f}"
    )
    print("候補:", ", ".join(f"{word}({score:.2f})" for word, score in result.candidates))


def temperature_label(similarity_to_target):
    if similarity_to_target >= 0.88:
        return "灼熱"
    if similarity_to_target >= 0.76:
        return "熱い"
    if similarity_to_target >= 0.62:
        return "温かい"
    if similarity_to_target >= 0.45:
        return "ぬるい"
    return "冷たい"


def run_scripted(space, seed, target, moves):
    rng = random.Random(seed)
    state = {"current": START_WORD, "target": target, "history": {START_WORD}, "turn": 1}
    for operation, ingredient, strength in moves:
        print_status(space, state)
        result = step(space, state, operation, ingredient, rng, strength)
        print_result(result)
        if state["turn"] > TURN_LIMIT:
            break
    print_status(space, state)


def run_interactive(space, seed, target):
    rng = random.Random(seed)
    state = {"current": START_WORD, "target": target, "history": {START_WORD}, "turn": 1}

    while state["turn"] <= TURN_LIMIT:
        print_status(space, state)
        raw = input("演算と素材語を入力 例: slerp 金 0.44 > ").strip()
        if raw in {"q", "quit", "exit"}:
            break
        parts = raw.split(maxsplit=2)
        if len(parts) < 2:
            print("入力形式: 演算 素材語 [強さ]")
            continue
        strength = float(parts[2]) if len(parts) == 3 else DEFAULT_STRENGTH
        result = step(space, state, parts[0], parts[1], rng, strength)
        print_result(result)

    print("\n終了")


def parse_moves(raw_moves):
    moves = []
    for raw in raw_moves:
        parts = raw.split(":")
        operation, ingredient = parts[:2]
        strength = float(parts[2]) if len(parts) >= 3 else DEFAULT_STRENGTH
        moves.append((operation, ingredient, strength))
    return moves


def main():
    parser = argparse.ArgumentParser(description="意味ベクトル錬金オークションの実験版")
    parser.add_argument("--mock", action="store_true", help="実モデルを使わずモックベクトルで遊ぶ")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--move", action="append", default=[], help="scripted move: operation:word[:strength]")
    args = parser.parse_args()

    space, source = load_space(use_mock=args.mock)
    if args.target not in space:
        print(f"目標語 {args.target} が語彙にないため {DEFAULT_TARGET} に切り替えます。")
        args.target = DEFAULT_TARGET
    print(f"vector source: {source}, seed: {args.seed}")

    if args.move:
        run_scripted(space, args.seed, args.target, parse_moves(args.move))
    else:
        run_interactive(space, args.seed, args.target)


if __name__ == "__main__":
    main()
