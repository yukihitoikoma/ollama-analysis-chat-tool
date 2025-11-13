import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_file(uploaded_file):
    """
    Load and process uploaded CSV file

    Args:
        uploaded_file: Streamlit file uploader object

    Returns:
        pandas.DataFrame: Loaded data or None if error
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Get basic information about the data
        file_info = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }

        logger.info(f"Successfully loaded CSV file: {uploaded_file.name}")
        return df, file_info

    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def get_file_info(df, file_info):
    """
    Display basic file information

    Args:
        df (pandas.DataFrame): Loaded data
        file_info (dict): File information dictionary
    """
    if df is not None:
        st.subheader("ファイル情報")
        st.write(f"ファイル名: {file_info['filename']}")
        st.write(f"サイズ: {file_info['size']} バイト")
        st.write(f"行数: {file_info['shape'][0]}")
        st.write(f"列数: {file_info['shape'][1]}")

        st.subheader("列情報")
        st.write(file_info['columns'])

        st.subheader("データ型")
        st.write(file_info['dtypes'])

def display_data_preview(df):
    """
    Display a preview of the data

    Args:
        df (pandas.DataFrame): Data to preview
    """
    if df is not None:
        st.subheader("データプレビュー")
        st.dataframe(df.head(10))

def save_uploaded_file(uploaded_file, save_directory):
    """
    Save uploaded file to specified directory

    Args:
        uploaded_file: Streamlit file uploader object
        save_directory (str): Directory to save file

    Returns:
        str: Path to saved file or None if error
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(save_directory, filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def load_csv(file_path):
    """CSVファイルを読み込む"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"CSVファイルの読み込みエラー: {str(e)}")

def save_dataframe(df, filename):
    """データフレームをCSVファイルとして保存"""
    try:
        # 日付フォルダを作成
        today = datetime.now().strftime("%Y%m%d")
        folder_path = os.path.join("data", today)
        os.makedirs(folder_path, exist_ok=True)

        # ファイルパスを構築
        file_path = os.path.join(folder_path, filename)

        # 重複ファイル名の処理
        counter = 1
        original_file_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_file_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1

        # データフレームを保存
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        raise Exception(f"ファイル保存エラー: {str(e)}")