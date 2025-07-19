# MiniRAG: 極めてシンプルな検索強化生成に向けて

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)

PostgreSQL + pgvector + Apache AGE の Docker はできたので、次は MiniRAG を Docker で構築してみる。
PostgreSQL16_pgvector(HNSW)_ApacheAGE_Docker に作っちゃったけど、最後には1つの Docker にまとめたい。


## MiniRAG の Docker

docker compose build --no-cache

docker compose up -d

docker compose down -v


http://localhost:8165/tree?


## MiniRAG クエリモード解説

MiniRAGは、検索拡張生成（RAG）のプロセスを最適化するために、複数のクエリモードを提供しています。各モードは、速度、コスト、および回答の品質のバランスが異なるため、ユースケースに応じて最適なものを選択できます。

### 1. NAIVEモード

**NAIVEモード**は、最もシンプルで直接的なRAGの手法です。

#### 仕組み

1.  ユーザーのクエリに基づいて、ベクトルデータベース（`chunks_vdb`）から関連性の高いテキストチャンクを検索します。
2.  取得したテキストチャンクをコンテキストとして大規模言語モデル（LLM）に提供します。
3.  LLMは、そのコンテキストに基づいて回答を生成します。

#### 特徴

*   **実装が容易**: 知識グラフなどの複雑なデータ構造を必要とせず、ベクトル検索のみに依存します。
*   **高速**: クエリ処理が単純なため、迅速に回答を得られます。
*   **低コスト**: 複雑な処理が不要なため、計算リソースの消費が少なくて済みます。

#### ユースケース

*   迅速な回答生成が求められる場合。
*   ドキュメントが構造化されておらず、エンティティや関係性の抽出が困難な場合。
*   シンプルなQ&Aシステム。

---

### 2. LIGHTモード

**LIGHTモード**は、`NAIVE`モードと`MINI`モードの中間に位置し、知識グラフを活用してより高品質なコンテキストを生成します。

#### 仕組み

1.  LLMを使用して、ユーザークエリから**高レベル**（概念、トピック）と**低レベル**（具体的なエンティティ名）のキーワードを抽出します。
2.  **高レベルキーワード**は、関係性ベクトルDB（`relationships_vdb`）を検索し、関連する関係性を特定します。
3.  **低レベルキーワード**は、エンティティベクトルDB（`entities_vdb`）を検索し、関連するエンティティを特定します。
4.  これらのエンティティと関係性から、知識グラフ上で関連するテキストチャンクや近傍エンティティを収集し、コンテキストを構築します。
5.  構築されたコンテキストをLLMに提供し、回答を生成します。

#### 特徴

*   **バランスの取れた性能**: `NAIVE`モードよりも高品質な回答を生成しつつ、`MINI`モードよりも高速に動作します。
*   **知識グラフの活用**: エンティティと関係性の両方を考慮することで、よりリッチなコンテキストを生成します。
*   **ハイブリッド検索**: キーワードのレベルに応じて異なる検索戦略を用いることで、効率的に情報を収集します。

#### ユースケース

*   ある程度の回答品質が求められ、かつリアルタイム性も重要な場合。
*   構造化されたデータと非構造化データが混在するドキュメントセット。
*   より複雑な質問に答える必要がある場合。

---

### 3. MINIモード

**MINIモード**は、最も高度で包括的なRAGの手法であり、知識グラフを最大限に活用して、深い推論に基づいた回答を生成します。

#### 仕組み

1.  LLMを使用して、ユーザークエリから**回答の期待される型**（例：「人名」「組織名」）と**クエリ内のエンティティ**を抽出します。
2.  抽出されたエンティティを基点に、知識グラフ上で**kホップ**内の近傍ノードを探索し、推論パスの候補を洗い出します。
3.  クエリとの関連性や、期待される回答の型に合致するノードへのパスを評価し、スコアリングします。
4.  最もスコアの高い推論パスに関連するエンティティ、関係性、テキストチャンクを収集し、詳細なコンテキストを構築します。
5.  構築されたコンテキストをLLMに提供し、回答を生成します。

#### 特徴

*   **最高の回答品質**: 知識グラフ上の多段階の推論（Multi-hop reasoning）により、直接的な情報だけでなく、間接的な関係性も考慮した深い回答を生成できます。
*   **詳細なコンテキスト**: 最も関連性の高い情報に絞り込むため、ノイズの少ない、質の高いコンテキストをLLMに提供できます。
*   **複雑なクエリへの対応**: 「AとBの関係は？」といった、複数のエンティティや関係性をまたぐ複雑な質問に効果的です。

#### ユースケース

*   最高の回答精度が求められる場合。
*   金融分析、科学研究、医療診断など、専門的な知識を要する分野。
*   複雑な因果関係や時系列の分析が必要な場合。

---

### まとめ

| モード | 特徴 | 長所 | 短所 |
| --- | --- | --- | --- |
| **NAIVE** | ベクトル検索のみ | 高速、低コスト、シンプル | 回答の質が低い可能性がある |
| **LIGHT** | エンティティと関係性のハイブリッド検索 | バランスの取れた性能 | NAIVEより遅く、MINIより質が低い |
| **MINI** | 知識グラフ上での多段階推論 | 最高の回答品質、複雑なクエリに対応 | 最も遅く、計算コストが高い |

---

コードリポジトリ: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**
<br />

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />
</div>

<a href='https://arxiv.org/abs/2501.06713'><img src='https://img.shields.io/badge/arXiv-2501.06713-b31b1b'>


## 🌍 READMEの翻訳

[English](./README.md) | [中文](./README_CN.md)

## 🎉 News
- [x] [2025.02.27]🎯📢`pip install minirag-hku`を使用して私たちのコードを実行できるようになりました！
- [x] [2025.02.14]🎯📢MiniRAGがNeo4j、PostgreSQL、TiDBなど10以上の異種グラフデータベースをサポートするようになりました。バレンタインデーおめでとう！🌹🌹🌹
- [x] [2025.02.05]🎯📢私たちのチームは、非常に長いコンテキストの動画を理解するVideoRAGをリリースしました。
- [x] [2025.02.01]🎯📢MiniRAGがAPI&Dockerデプロイメントをサポートするようになりました。詳細はこちらをご覧ください。

## TLDR
MiniRAGは、異種グラフインデックスと軽量なトポロジー強化検索を通じて、小さなモデルでも優れたRAGパフォーマンスを実現する極めてシンプルな検索強化生成フレームワークです。

## 概要
効率的で軽量な検索強化生成（RAG）システムの需要が高まる中、既存のRAGフレームワークに小型言語モデル（SLM）を導入する際の重大な課題が浮き彫りになっています。現在のアプローチは、SLMの限られた意味理解とテキスト処理能力のために深刻な性能低下に直面しており、リソースが限られたシナリオでの広範な採用に障害をもたらしています。これらの根本的な制限に対処するために、私たちは極めてシンプルで効率的な新しいRAGシステムである**MiniRAG**を提案します。**MiniRAG**は、2つの重要な技術革新を導入しています：（1）テキストチャンクと名前付きエンティティを統一構造に組み合わせる意味認識異種グラフインデックスメカニズム、これにより複雑な意味理解への依存を減らします。（2）高度な言語能力を必要とせずにグラフ構造を活用して効率的な知識発見を実現する軽量なトポロジー強化検索アプローチ。私たちの広範な実験は、**MiniRAG**がSLMを使用してもLLMベースの方法と同等の性能を達成しながら、ストレージスペースの25％しか必要としないことを示しています。さらに、複雑なクエリを持つ現実的なオンデバイスシナリオで軽量RAGシステムを評価するための包括的なベンチマークデータセットLiHua-Worldも提供します。

## MiniRAGフレームワーク

![MiniRAG](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

MiniRAGは、異種グラフインデックスと軽量なグラフベースの知識検索という主要なコンポーネントに基づいて構築された簡素化されたワークフローを採用しています。このアーキテクチャは、オンデバイスRAGシステムが直面する独自の課題に対処し、効率と効果の両方を最適化します。


## インストール

* ソースからインストール（推奨）

```bash
cd MiniRAG
pip install -e .
```
* PyPIからインストール（私たちのコードは[LightRAG](https://github.com/HKUDS/LightRAG)に基づいているため、直接インストールできます）

```bash
pip install lightrag-hku
```

## クイックスタート
* すべてのコードは`./reproduce`にあります。
* 必要なデータセットをダウンロードします。
* データセットを`./dataset`ディレクトリに配置します。
* 注：LiHua-Worldデータセットは`./dataset/LiHua-World/data/`に`LiHuaWorld.zip`として既に配置されています。他のデータセットを使用したい場合は、`./dataset/xxx`に配置できます。


次に、以下のbashコマンドを使用してデータセットをインデックスします：
```bash
python ./reproduce/Step_0_index.py
python ./reproduce/Step_1_QA.py
```

または、`./main.py`のコードを使用してMiniRAGを初期化します。


### 全体のパフォーマンステーブル
| モデル | NaiveRAG | | GraphRAG | | LightRAG | | **MiniRAG** | |
|-------|----------|----------|-----------|----------|-----------|----------|----------|----------|
| | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ |
| LiHua-World | | | | | | | | |
| Phi-3.5-mini-instruct | 41.22% | 23.20% | / | / | 39.81% | 25.39% | **53.29%** | 23.35% |
| GLM-Edge-1.5B-Chat | 42.79% | 24.76% | / | / | 35.74% | 25.86% | **52.51%** | 25.71% |
| Qwen2.5-3B-Instruct | 43.73% | 24.14% | / | / | 39.18% | 28.68% | **48.75%** | 26.02% |
| MiniCPM3-4B | 43.42% | 17.08% | / | / | 35.42% | 21.94% | **51.25%** | 21.79% |
| gpt-4o-mini | 46.55% | 19.12% | 35.27% | 37.77% | **56.90%** | 20.85% | 54.08% | 19.44% |
| MultiHop-RAG | | | | | | | | |
| Phi-3.5-mini-instruct | 42.72% | 31.34% | / | / | 27.03% | 11.78% | **49.96%** | 28.44% |
| GLM-Edge-1.5B-Chat | 44.44% | 24.26% | / | / | / | / | **51.41%** | 23.44% |
| Qwen2.5-3B-Instruct | 39.48% | 31.69% | / | / | 21.91% | 13.73% | **48.55%** | 33.10% |
| MiniCPM3-4B | 39.24% | 31.42% | / | / | 19.48% | 10.41% | **47.77%** | 26.88% |
| gpt-4o-mini | 53.60% | 27.19% | 60.92% | 16.86% | 64.91% | 19.37% | **68.43%** | 19.41% |


表中、/はその方法が効果的な応答を生成するのに苦労していることを意味します。

## 再現
すべてのコードは`./reproduce`ディレクトリにあります。

## コード構造

```python
├── dataset
│   └── LiHua-World
│       ├── README.md
│       ├── README_CN.md
│       ├── data
│       │   ├── LiHuaWorld.zip
│       └── qa
│           ├── query_set.csv
│           └── query_set.json
├── minirag
│   ├── kg
│   │   ├── __init__.py
│   │   ├── neo4j_impl.py
│   │   └── oracle_impl.py
│   ├── __init__.py
│   ├── base.py
│   ├── llm.py
│   ├── minirag.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.py
├── reproduce
│   ├── Step_0_index.py
│   └── Step_1_QA.py
├── LICENSE
├── main.py
├── README.md
├── README_CN.md
├── requirements.txt
├── setup.py
```

## データセット: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-Worldは、仮想ユーザーLiHuaの1年間のチャット記録を含む、オンデバイスRAGシナリオ専用に設計されたデータセットです。このデータセットには、シングルホップ、マルチホップ、およびサマリーの3種類の質問が含まれており、各質問には手動で注釈が付けられた回答とサポート文書がペアになっています。詳細については、[LiHua-WorldデータセットのREADME](./dataset/LiHua-World/README.md)を参照してください。


## Star History

<a href="https://star-history.com/#HKUDS/MiniRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
 </picture>
</a>

## Contribution

MiniRAGプロジェクトのすべての貢献者に感謝します！

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>

## 謝辞
私たちのフレームワークとコードリポジトリの基礎となる関連作業については、[nano-graphrag](https://github.com/gusye1234/nano-graphrag)および[LightRAG](https://github.com/HKUDS/LightRAG)を参照してください。素晴らしい仕事に感謝します。

## 🌟引用

```python
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

**私たちの仕事に興味を持っていただき、ありがとうございます！**