# alchemy_py

単語ベクトルを使った、りんごから目標単語へ近づけていく合成ゲームです。

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

この環境では `.venv` を Python 3.12.14 で作成済みです。Python 3.14 では `gensim` のビルドが失敗するため、Python 3.12 または 3.13 系を使ってください。

## Run

```bash
source .venv/bin/activate
python game.py
```

`game.py` は `model/entity_vector.kv` を読み込みます。モデルファイルがない場合は、`model/entity_vector.model.txt` を用意してから `convert.py` で変換してください。

```bash
source .venv/bin/activate
python convert.py
```

## Experimental Game

`semantic_alchemy.py` は既存ゲームから切り離した実験版です。実モデルが読み込める場合は `model/entity_vector.kv` を使い、足りない場合はモックベクトルへ自動で切り替わります。

```bash
source .venv/bin/activate
python semantic_alchemy.py --mock --seed 3
```

自動実行の例:

```bash
python semantic_alchemy.py --mock --seed 3 \
  --move slerp:金:0.44 \
  --move repel:借金:0.62 \
  --move purify:夢:0.25
```

入力形式は `演算 素材語 [強さ]` です。強さは `0.01` から `1.0` までで、指定しない場合は `0.44` です。

- `mix`: 安定した重み付き合成
- `slerp`: 球面線形補間で意味の表面をなめらかに移動
- `subtract`: 素材語の方向を引いて意外性を狙う
- `repel`: 目標へ寄せつつ危険語から反発
- `purify`: 現在語に対する直交成分を抽出して新しい軸を足す

結果には落札価格と、目標への近さ、意味の整合性、希少性、意外性、危険度の内訳が表示されます。

`seed` は候補単語の中から採用する単語をランダムに選ぶための乱数種です。同じ seed、同じ目標語、同じ手順なら結果を再現しやすくなります。

ゲーム演出として、目標語に近づく手を続けるとコンボが伸び、落札価格に倍率がかかります。大きく近づいた手や危険度の高い手には `Goal Rush`、`Wild Vector`、`High Roller` などの称号が付き、Web UI のログにも残ります。Web UI では `現在語 + 素材語 -> 素材後` の形でターン展開も確認できます。

## Web UI / API

FastAPI の薄いAPIと、出力確認用の最小Web UIを追加しています。Swagger UI は FastAPI 標準の `/docs` で使えます。

```bash
source .venv/bin/activate
uvicorn app:app --reload
```

- Web UI: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

主なAPI:

- `POST /api/games`: 目標語とseedを指定してゲーム開始
- `GET /api/games/{game_id}`: 現在状態を取得
- `POST /api/games/{game_id}/steps`: 演算、素材語、強さを送って1手進める
- `GET /api/operations`: 演算一覧
- `GET /api/vocabulary`: 語彙候補

Next.js や React Native からはこのJSON APIをそのまま呼び出せます。まずはWeb UIでレスポンス形状とゲーム感を確認し、画面を本格化するときにフロントエンドを別プロジェクト化する想定です。

## Test

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

## Design Ideas

- 錬金術の作業台風UI: 現在の単語を素材、入力単語を触媒、変化後の単語を生成物として表示する。
- 類似度メーター: ゴールに近づくほど色と音を変え、0.72 以上でフラスコが発光する。
- 単語の軌跡マップ: これまで出た単語をノードでつなぎ、意味空間を探索している感覚を出す。
- 失敗も演出する: 辞書にない単語や候補なしのときは煙、割れた瓶、ノイズなどで返す。
- モード追加: ターン制限、ランダム目標、デイリーチャレンジ、友達が作った目標を当てるモード。
