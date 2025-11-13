import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Import file storage utilities
from utils.file_storage import save_file_with_timestamp, save_chat_session, save_analysis_results, save_visualization

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
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

# Function to get LLM response
def get_llm_response(prompt, data_summary, model_type="ollama"):
    if model_type == "ollama":
        return get_ollama_response(prompt, data_summary)
    else:
        return get_claude_response(prompt, data_summary)

def get_ollama_response(prompt, data_summary):
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "192.168.0.103:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

        url = f"http://{ollama_host}/api/generate"

        # Prepare the prompt
        full_prompt = f"データの概要:\n{data_summary}\n\n質問:\n{prompt}\n\n回答:"

        payload = {
            "model": ollama_model,
            "prompt": full_prompt,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=30000)
        response.raise_for_status()

        result = response.json()
        return result["response"]

    except Exception as e:
        return f"Ollamaエラー: {str(e)}"

def get_claude_response(prompt, data_summary):
    try:
        claude_api_key = os.getenv("CLAUDE_API")
        claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4")
        if not claude_api_key:
            return "Claude APIキーが設定されていません"

        url = "https://api.anthropic.com/v1/messages"

        # Prepare the prompt
        full_prompt = f"データの概要:\n{data_summary}\n\n質問:\n{prompt}\n\n回答:"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": claude_model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": full_prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30000)
        response.raise_for_status()

        result = response.json()
        return result["content"][0]["text"]

    except Exception as e:
        return f"Claudeエラー: {str(e)}"

# Function to create data summary for LLM
def create_data_summary(data):
    """
    Create a summary of the data for LLM input

    Args:
        data (pandas.DataFrame): The data to summarize

    Returns:
        str: Data summary
    """
    if data is None:
        return "データがありません"

    summary = f"""データの概要:
- 行数: {len(data)}
- 列数: {len(data.columns)}
- 列名: {', '.join(data.columns)}
- データ型:
"""

    for col, dtype in data.dtypes.items():
        summary += f"  - {col}: {dtype}\n"

    summary += "\n基本統計情報:\n"
    summary += str(data.describe())

    return summary

# Function to create and save visualization
def create_and_save_visualization(df, question):
    """
    Create a visualization based on the question and save it

    Args:
        df (pandas.DataFrame): The data to visualize
        question (str): The question that triggered the visualization

    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create a simple bar chart for demonstration
        # In a real implementation, this would be more sophisticated
        if len(df.columns) >= 2:
            # Use first two columns for visualization
            col1 = df.columns[0]
            col2 = df.columns[1]

            # Create a simple bar chart
            fig = px.bar(df, x=col1, y=col2, title=f"データの可視化: {col1} vs {col2}")

            # Save the visualization
            folder_path = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(folder_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{timestamp}.png"
            file_path = os.path.join(folder_path, filename)

            fig.write_image(file_path)

            return file_path
        else:
            # If we don't have enough columns, create a simple histogram
            fig = px.histogram(df, title="データの分布")
            folder_path = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(folder_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{timestamp}.png"
            file_path = os.path.join(folder_path, filename)

            fig.write_image(file_path)

            return file_path

    except Exception as e:
        st.error(f"可視化作成エラー: {str(e)}")
        return None

# Function to save chat session
def save_chat_session(messages, data_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_session_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# CSVデータ分析チャットセッション\n\n")
        for msg in messages:
            if msg["role"] == "user":
                f.write(f"**ユーザー:** {msg['content']}\n\n")
            elif msg["role"] == "assistant":
                f.write(f"**アシスタント:** {msg['content']}\n\n")

    return filename

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Save uploaded file to disk only once
        if "saved_file_path" not in st.session_state:
            folder_path = os.path.join("data", datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(folder_path, exist_ok=True)
            saved_file_path = save_file_with_timestamp(folder_path, uploaded_file.name, uploaded_file.getvalue())
            st.session_state.saved_file_path = saved_file_path
            st.info(f"ファイルが保存されました: {saved_file_path}")

        # Display file info
        st.subheader("ファイル情報")
        st.write(f"ファイル名: {uploaded_file.name}")
        st.write(f"サイズ: {uploaded_file.size} バイト")
        st.write(f"行数: {len(df)}")
        st.write(f"列数: {len(df.columns)}")

        # Display column info
        st.subheader("列情報")
        # Fix for the dtype display issue
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ['列名', 'データ型']
        # Convert data types to string to avoid Arrow conversion issues
        dtypes_df['データ型'] = dtypes_df['データ型'].astype(str)
        st.dataframe(dtypes_df)

        # Display data preview
        st.subheader("データプレビュー")
        st.dataframe(df.head(10))

        # Display basic statistics
        st.subheader("基本統計情報")
        st.dataframe(df.describe())

        # Display missing values
        st.subheader("欠損値")
        missing_data = df.isnull().sum()
        st.dataframe(missing_data[missing_data > 0])

        # Chat interface
        st.subheader("チャットインターフェース")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("データに関する質問を入力してください"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from LLM
            with st.chat_message("assistant"):
                # Create data summary for LLM
                data_summary = create_data_summary(st.session_state.df)
                response = get_llm_response(prompt, data_summary, st.session_state.model)

                # Create and save visualization if needed
                visualization_path = create_and_save_visualization(st.session_state.df, prompt)
                if visualization_path:
                    # Add a link to the visualization in the response
                    response_with_visualization = f"{response}\n\n![可視化結果]({visualization_path})"
                    st.markdown(response_with_visualization)
                else:
                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Save button
            if st.button("チャットセッションを保存"):
                filename = save_chat_session(st.session_state.messages, uploaded_file.name)
                st.success(f"チャットセッションが保存されました: {filename}")

    except Exception as e:
        st.error(f"ファイル処理エラー: {str(e)}")
else:
    st.info("CSVファイルをアップロードしてください")
