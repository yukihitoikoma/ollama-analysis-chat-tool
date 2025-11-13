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