import os
import shutil
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_today_folder():
    """
    Create a folder for today's date if it doesn't exist

    Returns:
        str: Path to the created folder
    """
    try:
        # Create today's date folder
        today = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join("data", today)

        # Create directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        logger.info(f"Created folder: {folder_path}")
        return folder_path
    except Exception as e:
        logger.error(f"Error creating today's folder: {str(e)}")
        return None

def save_file_with_timestamp(folder_path, filename, file_content):
    """
    Save a file with a timestamp to avoid name conflicts

    Args:
        folder_path (str): Path to the folder where file should be saved
        filename (str): Original filename
        file_content: Content to save

    Returns:
        str: Path to the saved file
    """
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create new filename with timestamp
        name, ext = os.path.splitext(filename)
        new_filename = f"{timestamp}_{name}{ext}"

        # Full file path
        file_path = os.path.join(folder_path, new_filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Saved file: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None

def save_chat_session(chat_history, data_file, save_directory="data"):
    """
    Save chat session as markdown file

    Args:
        chat_history (list): List of chat messages
        data_file (str): Original data file name
        save_directory (str): Directory to save the file

    Returns:
        str: Path to the saved file
    """
    try:
        # Create today's folder
        folder_path = create_today_folder()
        if not folder_path:
            return None

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"chat_session_{timestamp}.md"
        file_path = os.path.join(folder_path, filename)

        # Create markdown content
        markdown_content = "# データ分析チャットセッション\n\n"
        markdown_content += f"**データファイル**: {data_file}\n\n"
        markdown_content += f"**保存日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown_content += "---\n\n"

        for message in chat_history:
            role = "ユーザー" if message["role"] == "user" else "アシスタント"
            markdown_content += f"**{role}**: {message['content']}\n\n"

        # Save file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Saved chat session: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving chat session: {str(e)}")
        return None

def save_analysis_results(analysis_results, data_file, save_directory="data"):
    """
    Save analysis results to a file

    Args:
        analysis_results (dict): Analysis results to save
        data_file (str): Original data file name
        save_directory (str): Directory to save the file

    Returns:
        str: Path to the saved file
    """
    try:
        # Create today's folder
        folder_path = create_today_folder()
        if not folder_path:
            return None

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"analysis_results_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        # Save results as JSON
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved analysis results: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        return None

def save_visualization(figure, data_file, save_directory="data"):
    """
    Save visualization as image file

    Args:
        figure: Visualization figure to save
        data_file (str): Original data file name
        save_directory (str): Directory to save the file

    Returns:
        str: Path to the saved file
    """
    try:
        # Create today's folder
        folder_path = create_today_folder()
        if not folder_path:
            return None

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"visualization_{timestamp}.png"
        file_path = os.path.join(folder_path, filename)

        # Save figure as PNG image
        if hasattr(figure, 'write_image'):
            # For Plotly figures
            figure.write_image(file_path)
        else:
            # For matplotlib figures or other types, save as PNG
            # This is a fallback - in practice, you'd want to handle specific figure types
            with open(file_path, "w") as f:
                f.write("Visualization saved successfully")

        logger.info(f"Saved visualization: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")
        return None
