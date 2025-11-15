# CSVデータ分析チャットアプリ

このアプリケーションは、Streamlitフレームワークを使用して構築されたCSVデータ分析ツールです。ユーザーはCSVファイルをアップロードし、チャットインターフェースを通じてデータに関する質問を行うことができます。LLMモデル（OllamaまたはClaude）を使用して自然言語でデータ分析を行います。

## 機能

### データ処理
- CSVファイルのアップロードと読み込み
- データの基本統計情報表示
- データプレビューと欠損値確認
- 自動的な日付フォルダへのファイル保存

### チャット分析
- OllamaまたはClaude LLMモデルの選択
- 自然言語でのデータ分析質問
- 自動的なデータ可視化（グラフ生成）
- チャットセッションの保存（テキスト + 画像リンク）

### 高度な分析機能
- 統計分析（ロジスティック回帰、線形回帰、相関分析、クラスタリング）
- 機械学習モデルの作成（XGBoost、LightGBM）
- 自動特徴量エンジニアリング（Featuretools）
- 特徴量選択（Boruta）
- ハイパーパラメータチューニング（Optuna）

### 可視化
- Plotlyを使用したインタラクティブなグラフ
- 自動的なグラフ保存
- チャットセッションへの画像リンク埋め込み

## セットアップ手順

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd 2025-11-12\ -\ analysis_tool_v2
```

### 2. 環境変数の設定
`.env`ファイルをプロジェクトルートに作成し、以下の内容を設定します:

```env
OLLAMA_HOST=192.168.0.103:11434
OLLAMA_MODEL=qwen3:30b
CLAUDE_API=your-claude-api-key-here
CLAUDE_MODEL=claude-haiku-4-5-20251001
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. アプリケーションの実行
```bash
streamlit run src/app.py
```

アプリケーションは自動的にブラウザで開きます（通常は http://localhost:8501）。

## 使用方法

### 基本的な使い方

1. **LLMモデルの選択**
   - アプリケーション上部でOllamaまたはClaudeを選択

2. **CSVファイルのアップロード**
   - 「CSVファイルをアップロードしてください」ボタンからCSVファイルを選択
   - ファイルは自動的に`data/YYYY-MM-DD/`フォルダに保存されます

3. **データ情報の確認**
   - ファイル情報（名前、サイズ、行数、列数）
   - 列名とデータ型
   - データプレビュー（最初の10行）
   - 基本統計情報
   - 欠損値の確認

4. **チャット分析**
   - チャット入力欄にデータに関する質問を入力
   - 例: 「このデータの相関関係を教えてください」
   - 例: 「年齢と収入の関係を可視化してください」
   - LLMが自然言語で回答し、必要に応じてグラフを生成

5. **チャットセッションの保存**
   - 「チャットセッションを保存」ボタンをクリック
   - チャットログと画像リンクがMarkdownファイルとして保存されます
   - 保存先: `data/YYYY-MM-DD/chat_session_YYYYMMDD_HHMMSS.md`

### 高度な分析

チャットで以下のような分析をリクエストできます:
- 統計分析（回帰分析、相関分析、クラスタリング）
- 機械学習モデルの作成と予測
- データの可視化とグラフ生成

### データファイルの管理

- アップロードされたCSVファイル: `data/YYYY-MM-DD/YYYYMMDD_HHMMSS_filename.csv`
- 生成されたグラフ: `data/YYYY-MM-DD/visualization_YYYYMMDD_HHMMSS.png`
- チャットセッション: `data/YYYY-MM-DD/chat_session_YYYYMMDD_HHMMSS.md`
- 機械学習モデル: `models/model_YYYYMMDD_HHMMSS.pkl`

## ディレクトリ構造

```
.
├── src/                          # ソースコード
│   ├── app.py                   # メインアプリケーション
│   ├── utils/                   # ユーティリティモジュール
│   │   ├── csv_handler.py      # CSV処理
│   │   ├── file_storage.py     # ファイル保存管理
│   │   └── llm_handler.py      # LLM API処理
│   ├── analysis/                # データ分析モジュール
│   │   ├── data_analyzer.py    # データ分析クラス
│   │   └── visualization.py    # 可視化機能
│   └── models/                  # 機械学習モジュール
│       └── ml_model.py         # ML モデル作成・管理
├── data/                        # データ保存ディレクトリ
│   └── YYYY-MM-DD/             # 日付別フォルダ（自動作成）
│       ├── *.csv               # アップロードされたCSVファイル
│       ├── visualization_*.png # 生成されたグラフ
│       └── chat_session_*.md   # チャットセッション
├── models/                      # 学習済みモデル保存ディレクトリ
│   └── model_*.pkl             # 保存されたモデル
├── docs/                        # ドキュメント
├── .env                         # 環境変数設定（要作成）
├── requirements.txt             # Python依存関係
├── speckit.constitution         # コード品質ガイドライン
├── speckit.specify             # アプリケーション仕様
└── README.md                   # このファイル
```

## 依存関係

### 主要フレームワーク
- **Streamlit** - Webアプリケーションフレームワーク
- **Python-dotenv** - 環境変数管理

### データ処理
- **Pandas** - データ操作
- **NumPy** - 数値計算

### 可視化
- **Matplotlib** - 基本的なグラフ作成
- **Seaborn** - 統計的データ可視化
- **Plotly** - インタラクティブグラフ

### 機械学習・統計分析
- **Scikit-learn** - 機械学習アルゴリズム
- **SciPy** - 科学計算
- **XGBoost** - 勾配ブースティング
- **LightGBM** - 軽量勾配ブースティング
- **TensorFlow** - ディープラーニング

### 特徴量エンジニアリング・最適化
- **Featuretools** - 自動特徴量エンジニアリング
- **Boruta** - 特徴量選択
- **Optuna** - ハイパーパラメータチューニング

### LLM統合
- **Requests** - HTTP API通信（Ollama/Claude API）

詳細は`requirements.txt`を参照してください。

## 技術スタック

- **言語**: Python 3.8+
- **UI**: Streamlit
- **LLM**: Ollama (qwen3:30b) / Claude (Haiku 4.5)
- **データベース**: ファイルベース（CSV、Pickle）
- **可視化**: Plotly, Matplotlib, Seaborn

## トラブルシューティング

### Ollamaに接続できない
- `.env`ファイルの`OLLAMA_HOST`設定を確認
- Ollamaサーバーが起動しているか確認
- ネットワーク接続を確認

### Claude APIエラー
- `.env`ファイルの`CLAUDE_API`キーが正しいか確認
- APIキーの有効期限を確認
- レート制限に達していないか確認

### グラフが保存されない
- `kaleido`パッケージがインストールされているか確認
  ```bash
  pip install kaleido
  ```

### モデル作成エラー
- データに欠損値が多すぎる場合は前処理が必要
- ターゲット変数が適切に設定されているか確認

## コード品質ガイドライン

このプロジェクトは以下の原則に従っています:
- PEP8準拠
- Google Style Docstrings
- 単一責任原則（SRP）
- DRY原則（重複排除）
- モジュラー設計

詳細は`speckit.constitution`を参照してください。

## ライセンス

このプロジェクトはMITライセンスの下で提供されます。

## 貢献

バグ報告や機能リクエストはIssueで受け付けています。
プルリクエストも歓迎します。

## サポート

質問や問題がある場合は、Issueを作成してください。