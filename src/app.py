"""CSV Data Analysis Chat Application.

This application allows users to upload CSV files and ask questions about
the data through a chat interface. It uses LLM models for natural language
processing and provides both textual and visual responses.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import file storage utilities
from utils.file_storage import (
    save_file_with_timestamp,
    save_chat_session,
    save_analysis_results,
    save_visualization
)

# Import LLM handler utilities
from utils.llm_handler import (
    get_llm_response,
    create_data_summary,
    detect_analysis_request,
    format_analysis_response,
    get_analysis_interpretation
)

# Import analysis utilities
from analysis.data_analyzer import DataAnalyzer
from analysis.visualization import (
    create_scatter_plot,
    create_correlation_heatmap,
    create_histogram,
    create_regression_plot,
    create_feature_importance_plot,
    create_clustering_plot,
    save_figure_to_file
)

# Import model utilities
from models.ml_model import (
    perform_statistical_analysis,
    prepare_data_for_modeling,
    create_advanced_ml_model,
    save_model
)

# Set page config
st.set_page_config(
    page_title="CSVデータ分析チャットアプリ",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = "ollama"  # Default to Ollama
if "visualizations" not in st.session_state:
    st.session_state.visualizations = []

# Title
st.title("📊 CSVデータ分析チャットアプリ")

# Model selection
model_choice = st.radio(
    "LLMモデルを選択:",
    ("Ollama", "Claude"),
    index=0 if st.session_state.model == "ollama" else 1
)
st.session_state.model = "ollama" if model_choice.startswith("Ollama") else "claude"

# File uploader
uploaded_file = st.file_uploader(
    "CSVファイルをアップロードしてください",
    type=["csv"]
)


def create_and_save_visualization(df, visualization_type="auto", analysis_results=None):
    """データに基づいて可視化を作成し保存する

    Args:
        df (pandas.DataFrame): 可視化するデータ
        visualization_type (str): 可視化のタイプ
        analysis_results (dict): 分析結果（オプション）

    Returns:
        tuple: (figure, file_path) 作成された図と保存先パス
    """
    try:
        fig = None

        # 可視化タイプに基づいてグラフを作成
        if visualization_type == "scatter":
            fig = create_scatter_plot(df)
        elif visualization_type == "correlation":
            fig = create_correlation_heatmap(df)
        elif visualization_type == "histogram":
            fig = create_histogram(df)
        elif visualization_type == "regression" and analysis_results:
            # 回帰分析結果がある場合
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = create_regression_plot(
                    df,
                    numeric_cols[0],
                    numeric_cols[1],
                    predictions=analysis_results.get("predictions")
                )
        elif visualization_type == "feature_importance" and analysis_results:
            # 特徴量重要度がある場合
            if "feature_importance" in analysis_results and "feature_names" in analysis_results:
                fig = create_feature_importance_plot(
                    analysis_results["feature_names"],
                    analysis_results["feature_importance"]
                )
        elif visualization_type == "clustering" and analysis_results:
            # クラスタリング結果がある場合
            if "cluster_labels" in analysis_results:
                fig = create_clustering_plot(df, analysis_results["cluster_labels"])
        else:
            # 自動選択：相関ヒートマップを優先
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = create_correlation_heatmap(df)
            elif len(numeric_cols) == 1:
                fig = create_histogram(df)

        if fig is None:
            return None, None

        # 保存先フォルダを作成
        folder_path = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
        file_path = save_figure_to_file(fig, folder_path)

        return fig, file_path

    except Exception as e:
        st.error(f"可視化作成エラー: {str(e)}")
        return None, None


def display_file_info(df, uploaded_file):
    """Display file information section.

    Args:
        df (pandas.DataFrame): The loaded data.
        uploaded_file: Streamlit file uploader object.
    """
    st.subheader("ファイル情報")
    st.write(f"ファイル名: {uploaded_file.name}")
    st.write(f"サイズ: {uploaded_file.size} バイト")
    st.write(f"行数: {len(df)}")
    st.write(f"列数: {len(df.columns)}")


def display_column_info(df):
    """Display column information section.

    Args:
        df (pandas.DataFrame): The loaded data.
    """
    st.subheader("列情報")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ['列名', 'データ型']
    dtypes_df['データ型'] = dtypes_df['データ型'].astype(str)
    st.dataframe(dtypes_df)


def display_data_preview(df):
    """Display data preview section.

    Args:
        df (pandas.DataFrame): The loaded data.
    """
    st.subheader("データプレビュー")
    st.dataframe(df.head(10))


def display_basic_statistics(df):
    """Display basic statistics section.

    Args:
        df (pandas.DataFrame): The loaded data.
    """
    st.subheader("基本統計情報")
    st.dataframe(df.describe())


def display_missing_values(df):
    """Display missing values section.

    Args:
        df (pandas.DataFrame): The loaded data.
    """
    st.subheader("欠損値")
    missing_data = df.isnull().sum()
    st.dataframe(missing_data[missing_data > 0])


def handle_chat_interaction(df, uploaded_file):
    """チャットインターフェースとユーザーインタラクションを処理する

    Args:
        df (pandas.DataFrame): 読み込まれたデータ
        uploaded_file: Streamlitファイルアップローダーオブジェクト
    """
    st.subheader("チャットインターフェース")

    # チャットメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 画像がある場合は表示
            if "image" in message:
                st.plotly_chart(message["image"], use_container_width=True)

    # チャット入力
    if prompt := st.chat_input("データに関する質問を入力してください"):
        # ユーザーメッセージをチャット履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)

        # アシスタントの応答を処理
        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                # データサマリーを作成
                data_summary = create_data_summary(df)

                # 分析リクエストを検出
                analysis_info = detect_analysis_request(prompt)

                # 分析結果を格納する辞書
                analysis_results = {}
                analysis_text = ""

                # 統計分析を実行
                if analysis_info["statistical_analysis"]:
                    try:
                        stat_results = perform_statistical_analysis(
                            df,
                            analysis_info["statistical_analysis"]
                        )

                        if "error" not in stat_results:
                            # 分析結果をテキスト化
                            if analysis_info["statistical_analysis"] == "logistic_regression":
                                analysis_text += "\n\n### ロジスティック回帰分析結果\n\n"
                                analysis_text += f"- **精度 (Accuracy):** {stat_results['metrics']['accuracy']:.4f}\n"
                                if stat_results.get("feature_importance") is not None:
                                    analysis_results["feature_importance"] = stat_results["feature_importance"]
                                    analysis_results["feature_names"] = [f"特徴量 {i}" for i in range(len(stat_results["feature_importance"]))]

                                    # 特徴量係数の表示
                                    analysis_text += "\n**特徴量係数（重み）:**\n\n"
                                    for i, coef in enumerate(stat_results["feature_importance"]):
                                        analysis_text += f"{i+1}. 特徴量 {i}: {coef:.6f}\n"

                            elif analysis_info["statistical_analysis"] == "linear_regression":
                                analysis_text += "\n\n### 重回帰分析結果\n\n"
                                analysis_text += f"- **平均二乗誤差 (MSE):** {stat_results['metrics']['mse']:.4f}\n"
                                analysis_text += f"- **平方根平均二乗誤差 (RMSE):** {stat_results['metrics']['rmse']:.4f}\n"
                                analysis_text += f"- **決定係数 (R²):** {stat_results['metrics']['r2']:.4f}\n"
                                if stat_results.get("feature_importance") is not None:
                                    analysis_results["feature_importance"] = stat_results["feature_importance"]
                                    analysis_results["feature_names"] = [f"特徴量 {i}" for i in range(len(stat_results["feature_importance"]))]

                                    # 特徴量係数の表示
                                    analysis_text += "\n**特徴量係数（重み）:**\n\n"
                                    for i, coef in enumerate(stat_results["feature_importance"]):
                                        analysis_text += f"{i+1}. 特徴量 {i}: {coef:.6f}\n"

                            elif analysis_info["statistical_analysis"] == "association_analysis":
                                analysis_text += "\n\n### 関連性分析結果\n\n"
                                analysis_text += "**強い相関関係 (|r| > 0.7):**\n\n"
                                for corr in stat_results.get("strong_correlations", []):
                                    analysis_text += f"- {corr['feature1']} と {corr['feature2']}: {corr['correlation']:.4f}\n"

                            elif analysis_info["statistical_analysis"] == "clustering":
                                analysis_text += "\n\n### クラスタリング分析結果\n\n"
                                analysis_text += f"- **クラスタ数:** {stat_results['n_clusters']}\n"
                                analysis_text += f"- **シルエットスコア:** {stat_results['silhouette_score']:.4f}\n"
                                analysis_results["cluster_labels"] = stat_results["cluster_labels"]

                            analysis_results["statistical_analysis"] = stat_results
                        else:
                            analysis_text += f"\n\n⚠️ {stat_results['error']}\n"
                    except Exception as e:
                        analysis_text += f"\n\n⚠️ 統計分析エラー: {str(e)}\n"

                # モデル作成を実行
                if analysis_info["model_creation"]:
                    try:
                        with st.spinner("モデルを作成中..."):
                            # データ準備
                            X, y, feature_names_list, target_encoder, scaler = prepare_data_for_modeling(df)

                            # 高度なモデル作成
                            model, metrics, feature_importance, feature_names = create_advanced_ml_model(
                                X, y,
                                model_type="xgboost",
                                target_encoder=target_encoder,
                                feature_selection_method="boruta",
                                hyperparameter_tuning=True
                            )

                            # モデル保存
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            model_path = f"models/model_{timestamp}.pkl"
                            os.makedirs("models", exist_ok=True)

                            # モデルのパラメータを取得
                            model_params = model.get_params() if hasattr(model, 'get_params') else {}

                            model_info = {
                                "model_type": "xgboost",
                                "metrics": metrics,
                                "feature_importance": feature_importance.tolist() if feature_importance is not None else None,
                                "feature_names": feature_names,
                                "model_params": model_params,
                                "timestamp": timestamp
                            }
                            save_model(model, model_path, model_info)

                            # 結果をテキスト化
                            analysis_text += "\n\n### 予測モデル作成結果\n\n"
                            analysis_text += f"- **モデルタイプ:** XGBoost\n"
                            analysis_text += f"- **特徴量選択:** Boruta\n"
                            analysis_text += f"- **ハイパーパラメータチューニング:** 有効\n"
                            analysis_text += f"- **モデル保存場所:** `{model_path}`\n\n"

                            # 最適化されたハイパーパラメータの表示
                            analysis_text += "**最適化されたハイパーパラメータ:**\n\n"
                            important_params = ['learning_rate', 'max_depth', 'n_estimators', 'subsample', 'colsample_bytree']
                            for param in important_params:
                                if param in model_params:
                                    analysis_text += f"- **{param}:** {model_params[param]}\n"

                            analysis_text += "\n**評価指標:**\n\n"
                            for metric, value in metrics.items():
                                analysis_text += f"- **{metric}:** {value:.4f}\n"

                            if feature_importance is not None:
                                analysis_text += "\n**特徴量重要度（全特徴量）:**\n\n"
                                # すべての特徴量を重要度順にソート
                                sorted_idx = np.argsort(feature_importance)[::-1]
                                for i, idx in enumerate(sorted_idx):
                                    fname = feature_names[idx] if idx < len(feature_names) else f"特徴量 {idx}"
                                    analysis_text += f"{i+1}. {fname}: {feature_importance[idx]:.6f}\n"

                            # 特徴量重要度をグラフ用に保存
                            if feature_importance is not None:
                                analysis_results["feature_importance"] = feature_importance
                                analysis_results["feature_names"] = feature_names
                                analysis_results["model_params"] = model_params

                    except Exception as e:
                        analysis_text += f"\n\n⚠️ モデル作成エラー: {str(e)}\n"

                # LLM応答を取得
                llm_response = get_llm_response(prompt, data_summary, st.session_state.model)

                # 完全な応答を構築
                full_response = llm_response + analysis_text

                # 分析結果がある場合は、LLMによる解釈を追加
                if analysis_text.strip():
                    with st.spinner("分析結果を解釈中..."):
                        interpretation = get_analysis_interpretation(
                            prompt,
                            data_summary,
                            analysis_text,
                            st.session_state.model
                        )
                        if interpretation:
                            full_response += "\n\n---\n\n"
                            full_response += "### 📊 分析結果の解釈と提案\n\n"
                            full_response += interpretation

                # 可視化を作成
                fig = None
                fig_path = None

                try:
                    if (analysis_info["visualization"] or
                        analysis_info["statistical_analysis"] or
                        analysis_info["model_creation"]):
                        # 可視化タイプを決定
                        viz_type = analysis_info["visualization"] or "auto"

                        # 特殊な可視化タイプ
                        if analysis_info["statistical_analysis"] == "linear_regression":
                            viz_type = "regression"
                        elif analysis_info["statistical_analysis"] == "clustering":
                            viz_type = "clustering"
                        elif (analysis_info["model_creation"] and
                              "feature_importance" in analysis_results):
                            viz_type = "feature_importance"

                        fig, fig_path = create_and_save_visualization(
                            df,
                            visualization_type=viz_type,
                            analysis_results=analysis_results
                        )

                        if fig_path:
                            logger.info(f"グラフ保存成功: {fig_path}")
                        else:
                            logger.warning("グラフの保存に失敗しました")
                except Exception as viz_error:
                    logger.error(f"可視化エラー: {str(viz_error)}")
                    st.warning(
                        "グラフの作成・保存中にエラーが発生しましたが、"
                        "分析結果は表示されます"
                    )

                # 応答を表示
                st.markdown(full_response)

                # グラフを表示
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)

                        # 画像リンクを追加
                        if fig_path:
                            full_response += f"\n\n![可視化結果]({fig_path})"
                    except Exception as display_error:
                        logger.error(f"グラフ表示エラー: {str(display_error)}")
                        st.error("グラフの表示に失敗しました")

                # メッセージをセッションに保存
                message_data = {"role": "assistant", "content": full_response}
                if fig:
                    message_data["image"] = fig
                st.session_state.messages.append(message_data)

                # 可視化パスを追跡
                if fig_path:
                    if "visualizations" not in st.session_state:
                        st.session_state.visualizations = []
                    st.session_state.visualizations.append(fig_path)

    # 保存ボタン
    if len(st.session_state.messages) > 0:
        if st.button("チャットセッションを保存"):
            try:
                filename = uploaded_file.name if uploaded_file else "unknown_file.csv"
                saved_file_path = save_chat_session(
                    st.session_state.messages,
                    filename
                )

                if saved_file_path:
                    st.success(f"チャットセッションが保存されました: {saved_file_path}")
                else:
                    st.error("チャットセッションの保存に失敗しました")
            except Exception as e:
                st.error(f"チャットセッションの保存に失敗しました: {str(e)}")


# Process uploaded file
if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Save uploaded file to disk only once
        if "saved_file_path" not in st.session_state:
            folder_path = os.path.join(
                "data",
                datetime.now().strftime("%Y-%m-%d")
            )
            os.makedirs(folder_path, exist_ok=True)
            saved_file_path = save_file_with_timestamp(
                folder_path,
                uploaded_file.name,
                uploaded_file.getvalue()
            )
            st.session_state.saved_file_path = saved_file_path
            st.info(f"ファイルが保存されました: {saved_file_path}")

        # Display file information sections
        display_file_info(df, uploaded_file)
        display_column_info(df)
        display_data_preview(df)
        display_basic_statistics(df)
        display_missing_values(df)

        # Handle chat interaction
        handle_chat_interaction(df, uploaded_file)

    except Exception as e:
        st.error(f"ファイル処理エラー: {str(e)}")
else:
    st.info("CSVファイルをアップロードしてください")
