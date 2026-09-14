from pathlib import Path


SOURCE_PATH = Path("model/entity_vector.model.txt")
OUTPUT_PATH = Path("model/entity_vector.kv")


def main():
    if not SOURCE_PATH.exists():
        print(f"{SOURCE_PATH} が見つかりません。変換元の txt モデルを配置してください。")
        return

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("gensim がインストールされていません。仮想環境を有効化して pip install -r requirements.txt を実行してください。")
        return

    print("txtモデル読み込み中...")
    model = KeyedVectors.load_word2vec_format(str(SOURCE_PATH), binary=False)

    print("保存中（1回だけ）...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_PATH))

    print("完了！")


if __name__ == "__main__":
    main()
