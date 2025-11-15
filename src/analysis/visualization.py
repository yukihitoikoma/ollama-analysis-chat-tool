import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_basic_visualizations(data):
    """
    Create basic visualizations for the data

    Args:
        data (pandas.DataFrame): The data to visualize

    Returns:
        dict: Dictionary of visualization figures
    """
    if data is None or data.empty:
        return {}

    try:
        visualizations = {}

        # Get numeric columns for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) == 0:
            return visualizations

        # Histogram for first numeric column
        if len(numeric_columns) > 0:
            fig = px.histogram(data, x=numeric_columns[0], title=f"分布図: {numeric_columns[0]}")
            visualizations['histogram'] = fig

        # Scatter plot for first two numeric columns
        if len(numeric_columns) >= 2:
            fig = px.scatter(data, x=numeric_columns[0], y=numeric_columns[1],
                           title=f"散布図: {numeric_columns[0]} vs {numeric_columns[1]}")
            visualizations['scatter'] = fig

        # Correlation heatmap
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title="相関行列")
            visualizations['correlation'] = fig

        return visualizations

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        return {}

def create_advanced_visualizations(data):
    """
    Create advanced visualizations for the data

    Args:
        data (pandas.DataFrame): The data to visualize

    Returns:
        dict: Dictionary of advanced visualization figures
    """
    if data is None or data.empty:
        return {}

    try:
        visualizations = {}

        # Get numeric columns for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) == 0:
            return visualizations

        # Box plot for first numeric column
        if len(numeric_columns) > 0:
            fig = px.box(data, y=numeric_columns[0], title=f"箱ひげ図: {numeric_columns[0]}")
            visualizations['box'] = fig

        # Bar chart for categorical data (if any)
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_columns) > 0:
            fig = px.histogram(data, x=categorical_columns[0], title=f"カテゴリ分布: {categorical_columns[0]}")
            visualizations['bar'] = fig

        return visualizations

    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {str(e)}")
        return {}

def display_visualizations(visualizations, st_container):
    """
    Display visualizations in Streamlit

    Args:
        visualizations (dict): Dictionary of visualization figures
        st_container: Streamlit container to display in
    """
    if not visualizations:
        st_container.info("表示できる視覚化データがありません")
        return

    try:
        for name, fig in visualizations.items():
            if fig is not None:
                st_container.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error displaying visualizations: {str(e)}")
        st_container.error("視覚化の表示に失敗しました")

def create_summary_plot(data):
    """
    Create a summary plot showing key statistics

    Args:
        data (pandas.DataFrame): The data to plot

    Returns:
        plotly.graph_objects.Figure: Summary plot
    """
    if data is None or data.empty:
        return None

    try:
        # Get numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) == 0:
            return None

        # Create summary statistics
        stats = data[numeric_columns].describe()

        # Create a simple bar chart of means
        fig = go.Figure()

        for col in numeric_columns:
            fig.add_trace(go.Bar(x=[col], y=[data[col].mean()], name=col))

        fig.update_layout(
            title="各列の平均値",
            xaxis_title="列名",
            yaxis_title="平均値"
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating summary plot: {str(e)}")
        return None

def create_advanced_visualizations(data):
    """
    Create advanced visualizations for the data

    Args:
        data (pandas.DataFrame): The data to visualize

    Returns:
        dict: Dictionary of visualization figures
    """
    if data is None or data.empty:
        return {}

    try:
        visualizations = {}

        # Get numeric columns for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) == 0:
            return visualizations

        # Box plot for first numeric column
        if len(numeric_columns) > 0:
            fig = px.box(data, y=numeric_columns[0], title=f"箱ひげ図: {numeric_columns[0]}")
            visualizations['box'] = fig

        # Bar chart for categorical data (if any)
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_columns) > 0:
            fig = px.histogram(data, x=categorical_columns[0], title=f"カテゴリ分布: {categorical_columns[0]}")
            visualizations['bar'] = fig

        return visualizations

    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {str(e)}")
        return {}

def display_visualizations(visualizations, st_container):
    """
    Display visualizations in Streamlit

    Args:
        visualizations (dict): Dictionary of visualization figures
        st_container: Streamlit container to display in
    """
    if not visualizations:
        st_container.info("表示できる視覚化データがありません")
        return

    try:
        for name, fig in visualizations.items():
            if fig is not None:
                st_container.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error displaying visualizations: {str(e)}")
        st_container.error("視覚化の表示に失敗しました")

def create_summary_visualizations(data):
    """
    Create summary visualizations for the data

    Args:
        data (pandas.DataFrame): The data to visualize

    Returns:
        dict: Dictionary of visualization figures
    """
    if data is None or data.empty:
        return {}

    try:
        visualizations = {}

        # Get numeric columns for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) == 0:
            return visualizations

        # Create a combined figure with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f"分布図: {numeric_columns[0]}", f"箱ひげ図: {numeric_columns[0]}",
                           f"散布図: {numeric_columns[0]} vs {numeric_columns[1]}", "相関行列"),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )

        # Add histogram
        fig.add_trace(
            go.Histogram(x=data[numeric_columns[0]], name="Histogram"),
            row=1, col=1
        )

        # Add box plot
        fig.add_trace(
            go.Box(y=data[numeric_columns[0]], name="Box Plot"),
            row=1, col=2
        )

        # Add scatter plot
        if len(numeric_columns) >= 2:
            fig.add_trace(
                go.Scatter(x=data[numeric_columns[0]], y=data[numeric_columns[1]], mode='markers', name="Scatter"),
                row=2, col=1
            )

        # Add correlation heatmap
        if len(numeric_columns) > 1:
            fig.add_trace(
                go.Heatmap(z=data[numeric_columns].corr(), x=numeric_columns, y=numeric_columns, name="Correlation"),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=False, title_text="データ要約図")
        visualizations['summary'] = fig

        return visualizations

    except Exception as e:
        logger.error(f"Error creating summary visualizations: {str(e)}")
        return {}


def create_scatter_plot(data, x_col=None, y_col=None, title=None):
    """散布図を作成する

    Args:
        data (pandas.DataFrame): データ
        x_col (str, optional): X軸の列名。指定がなければ最初の数値列
        y_col (str, optional): Y軸の列名。指定がなければ2番目の数値列
        title (str, optional): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: 散布図
    """
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        if x_col is None:
            x_col = numeric_cols[0]
        if y_col is None:
            y_col = numeric_cols[1]

        if title is None:
            title = f"散布図: {x_col} vs {y_col}"

        fig = px.scatter(data, x=x_col, y=y_col, title=title)
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        return fig

    except Exception as e:
        logger.error(f"散布図作成エラー: {str(e)}")
        return None


def create_correlation_heatmap(data, title="相関行列ヒートマップ"):
    """相関行列のヒートマップを作成する

    Args:
        data (pandas.DataFrame): データ
        title (str): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: ヒートマップ
    """
    try:
        numeric_data = data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) < 2:
            return None

        corr_matrix = numeric_data.corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="相関係数"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title=title
        )

        fig.update_traces(text=corr_matrix.round(2).values, texttemplate="%{text}")
        return fig

    except Exception as e:
        logger.error(f"相関ヒートマップ作成エラー: {str(e)}")
        return None


def create_histogram(data, col=None, title=None):
    """ヒストグラムを作成する

    Args:
        data (pandas.DataFrame): データ
        col (str, optional): 列名。指定がなければ最初の数値列
        title (str, optional): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: ヒストグラム
    """
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return None

        if col is None:
            col = numeric_cols[0]

        if title is None:
            title = f"ヒストグラム: {col}"

        fig = px.histogram(data, x=col, title=title, nbins=30)
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        return fig

    except Exception as e:
        logger.error(f"ヒストグラム作成エラー: {str(e)}")
        return None


def create_regression_plot(data, x_col, y_col, predictions=None, title=None):
    """回帰分析のプロットを作成する

    Args:
        data (pandas.DataFrame): データ
        x_col (str): X軸の列名
        y_col (str): Y軸の列名
        predictions (array-like, optional): 予測値
        title (str, optional): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: 回帰プロット
    """
    try:
        if title is None:
            title = f"回帰分析: {x_col} vs {y_col}"

        fig = go.Figure()

        # 実データの散布図
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='markers',
            name='実データ',
            marker=dict(size=8, opacity=0.6)
        ))

        # 予測値がある場合は回帰線を追加
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=predictions,
                mode='lines',
                name='予測値',
                line=dict(color='red', width=2)
            ))

        fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
        return fig

    except Exception as e:
        logger.error(f"回帰プロット作成エラー: {str(e)}")
        return None


def create_feature_importance_plot(feature_names, importances, title="特徴量重要度"):
    """特徴量重要度のバーグラフを作成する

    Args:
        feature_names (list): 特徴量名のリスト
        importances (array-like): 特徴量重要度の配列
        title (str): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: バーグラフ
    """
    try:
        # 重要度順にソート
        sorted_idx = np.argsort(importances)[::-1][:10]  # 上位10個
        sorted_names = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in sorted_idx]
        sorted_importances = [importances[i] for i in sorted_idx]

        fig = go.Figure([
            go.Bar(
                x=sorted_importances,
                y=sorted_names,
                orientation='h',
                marker=dict(color='lightblue')
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="重要度",
            yaxis_title="特徴量",
            yaxis=dict(autorange="reversed")
        )
        return fig

    except Exception as e:
        logger.error(f"特徴量重要度プロット作成エラー: {str(e)}")
        return None


def create_clustering_plot(data, cluster_labels, title="クラスタリング結果"):
    """クラスタリング結果の散布図を作成する

    Args:
        data (pandas.DataFrame): データ
        cluster_labels (array-like): クラスタラベル
        title (str): グラフのタイトル

    Returns:
        plotly.graph_objects.Figure: クラスタリングプロット
    """
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        # PCAで2次元に削減（3次元以上の場合）
        if len(numeric_cols) > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(data[numeric_cols])
            x_col, y_col = "PC1", "PC2"
        else:
            coords = data[numeric_cols[:2]].values
            x_col, y_col = numeric_cols[0], numeric_cols[1]

        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=cluster_labels.astype(str),
            title=title,
            labels={'x': x_col, 'y': y_col, 'color': 'クラスタ'}
        )
        return fig

    except Exception as e:
        logger.error(f"クラスタリングプロット作成エラー: {str(e)}")
        return None


def save_figure_to_file(fig, folder_path, filename=None):
    """Plotlyの図をファイルに保存する

    Args:
        fig (plotly.graph_objects.Figure): Plotlyの図
        folder_path (str): 保存先フォルダ
        filename (str, optional): ファイル名。指定がなければタイムスタンプを使用

    Returns:
        str: 保存されたファイルのパス
    """
    try:
        import os
        from datetime import datetime

        if fig is None:
            logger.warning("図がNullのため保存できません")
            return None

        # フォルダが存在しない場合は作成
        os.makedirs(folder_path, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{timestamp}.png"

        file_path = os.path.join(folder_path, filename)

        # 画像として保存
        try:
            fig.write_image(file_path, width=1200, height=800)
            logger.info(f"グラフを保存しました: {file_path}")
            return file_path
        except ImportError as ie:
            logger.error(
                f"kaleido パッケージがインストールされていません: {str(ie)}"
            )
            logger.error("pip install kaleido を実行してください")
            return None
        except Exception as write_error:
            logger.error(f"グラフ書き込みエラー: {str(write_error)}")
            return None

    except Exception as e:
        logger.error(f"グラフ保存エラー: {str(e)}")
        return None