import random
from pathlib import Path


MODEL_PATH = Path("model/entity_vector.kv")
START_WORD = "りんご"
CLEAR_SIMILARITY = 0.72


def load_model(model_path=MODEL_PATH):
    from gensim.models import KeyedVectors

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} が見つかりません。"
            "model/entity_vector.model.txt を用意してから convert.py を実行してください。"
        )

    print("日本語モデルを読み込み中...（初回は2〜3分）")
    try:
        model = KeyedVectors.load(str(model_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"モデルを構成するファイルが足りません: {exc.filename}。"
            "convert.py で生成された .kv と付随する .npy を同じ場所に置いてください。"
        ) from exc
    print("読み込み完了！ゲーム開始！")
    return model


def similarity(model, word1, word2):
    if word1 not in model or word2 not in model:
        return None
    return float(model.similarity(word1, word2))


def combine_words(model, current_word, input_word, history, beta=0.35):
    import numpy as np

    if current_word not in model:
        return None, f"『{current_word}』は辞書にありません"
    if input_word not in model:
        return None, f"『{input_word}』は辞書にありません"

    v1 = model[current_word]
    v2 = model[input_word]

    # 「少しだけ近づく」遷移
    v_new = v1 + beta * (v2 - v1)

    # 正規化
    v_new = v_new / np.linalg.norm(v_new)

    # 候補取得
    candidates = model.similar_by_vector(v_new, topn=15)

    # 元単語を除外
    filtered = []

    for word, score in candidates:
        if word != current_word and word != input_word and word not in history:
            filtered.append((word, score))

    if not filtered:
        return None, "候補が見つかりません"

    new_word, score = random.choice(filtered)
    return new_word, score


def progress_message(sim):
    if sim is None:
        return "判定不能"
    if sim < 0.25:
        return "かなり遠い…"
    if sim < 0.45:
        return "少し関係ありそう"
    if sim < 0.60:
        return "近づいてきた！"
    if sim < CLEAR_SIMILARITY:
        return "かなり近い！！"
    return "到達圏内！！！"


def play(model):
    target_word = input("今回の目標単語を入力してください：").strip()

    if not target_word:
        print("目標単語が空です。終了します。")
        return

    if target_word not in model:
        print(f"『{target_word}』は辞書にありません。終了します。")
        return

    current = START_WORD
    history = {START_WORD}
    turn = 1

    print(f"\nスタート単語：{START_WORD}")
    print(f"ヒント：{len(target_word)}文字")

    while True:
        print(f"\n--- Turn {turn} ---")
        print("現在の単語:", current)

        sim = similarity(model, current, target_word)
        if sim is not None:
            print(f"目標との類似度: {sim:.3f}")
            print(progress_message(sim))

            if sim >= CLEAR_SIMILARITY:
                print("\nクリア！！")
                print("答え:", target_word)
                break
        else:
            print("目標単語との類似度を計算できません")

        player_input = input("入れる単語：").strip()
        if not player_input:
            print("単語を入力してください")
            continue

        new_word, info = combine_words(model, current, player_input, history)

        if new_word is None:
            print(info)
            continue

        print("→ 変化:", new_word)

        current = new_word
        history.add(new_word)
        turn += 1


def main():
    try:
        model = load_model()
    except FileNotFoundError as exc:
        print(exc)
        return
    except ImportError:
        print("gensim がインストールされていません。仮想環境を有効化して pip install -r requirements.txt を実行してください。")
        return
    except Exception as exc:
        print(f"モデルを読み込めませんでした: {exc}")
        return

    play(model)


if __name__ == "__main__":
    main()

# 今回の目標単語を入力してください：宇宙

# スタート単語：りんご
# 目標：？？？（ヒント：2文字）

# --- Turn 1 ---
# 現在の単語: りんご
# 目標への近さランキング：18234位
# 入れる単語：ロケット

# → 変化: NASA

# --- Turn 2 ---
# 現在の単語: NASA
# 目標への近さランキング：132位
# 入れる単語：星

# → 変化: 銀河

# --- Turn 3 ---
# 現在の単語: 銀河
# 目標への近さランキング：18位

# 🎉 クリア！目標に到達しました！ 🎉
# 答え: 宇宙
