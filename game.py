from gensim.models import KeyedVectors
import numpy as np
import random

# -----------------------------
# モデル読み込み
# -----------------------------
print("日本語モデルを読み込み中...（初回は2〜3分）")

# model = KeyedVectors.load_word2vec_format(
#     "model/entity_vector.model.bin",
#     binary=True
# )

# ※ 処理時間がだいぶ思いから使わない
# model = KeyedVectors.load_word2vec_format(
#     "model/entity_vector.txt",
#     binary=False
# )

# 高速化するならモデルを保存しておく
model = KeyedVectors.load("model/entity_vector.kv")

print("読み込み完了！ゲーム開始！")


# -----------------------------
# 距離（コサイン類似度）
# -----------------------------
def similarity(word1, word2):
    if word1 not in model or word2 not in model:
        return None
    return float(model.similarity(word1, word2))


# -----------------------------
# 単語合成（ゲームの核）
# -----------------------------
def combine_words(current_word, input_word, history, beta=0.35):

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

# -----------------------------
# 進捗表示（プレイヤーの体感用）
# -----------------------------
def progress_message(sim):
    if sim is None:
        return "判定不能"
    elif sim < 0.25:
        return "かなり遠い…"
    elif sim < 0.45:
        return "少し関係ありそう"
    elif sim < 0.60:
        return "近づいてきた！"
    elif sim < 0.72:
        return "かなり近い！！"
    else:
        return "到達圏内！！！"

# -----------------------------
# ゲーム設定
# -----------------------------
START_WORD = "りんご"
TARGET_WORD = input("今回の目標単語を入力してください：")

if TARGET_WORD not in model:
    print(None, f"『{TARGET_WORD}』は辞書にありません")

current = START_WORD
history = {START_WORD}

print(f"\nスタート単語：{START_WORD}")
print(f"ヒント：{len(TARGET_WORD)}文字")


# -----------------------------
# メインループ
# -----------------------------
turn = 1

while True:

    print(f"\n--- Turn {turn} ---")
    print("現在の単語:", current)

    # ゴール距離（重要：順位ではなく距離）
    sim = similarity(current, TARGET_WORD)

    if sim is not None:
        print(f"目標との類似度: {sim:.3f}")
        print(progress_message(sim))

        # クリア判定
        if sim >= 0.72:
            print("\n🎉 クリア！！ 🎉")
            print("答え:", TARGET_WORD)
            break
    else:
        print("※目標単語が辞書に無い可能性があります")

    player_input = input("入れる単語：")

    new_word, info = combine_words(current, player_input, history)

    if new_word is None:
        print(info)
        continue

    print("→ 変化:", new_word)

    current = new_word
    history.add(new_word)
    turn += 1

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