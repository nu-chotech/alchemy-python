from gensim.models import KeyedVectors

print("txtモデル読み込み中...")
model = KeyedVectors.load_word2vec_format(
    "model/entity_vector.model.txt",
    binary=False
)

print("保存中（1回だけ）...")
model.save("model/entity_vector.kv")

print("完了！")