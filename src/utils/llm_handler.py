import os
import logging
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm_response(prompt, data_summary, model_type="ollama"):
    """
    Get response from LLM based on the selected model

    Args:
        prompt (str): User's question
        data_summary (str): Summary of the data
        model_type (str): Type of model to use ("ollama" or "claude")

    Returns:
        str: Response from the LLM
    """
    try:
        if model_type == "ollama":
            return get_ollama_response(prompt, data_summary)
        elif model_type == "claude":
            return get_claude_response(prompt, data_summary)
        else:
            return "Unknown model type"
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return f"LLMエラー: {str(e)}"

def get_ollama_response(prompt, data_summary):
    """
    Get response from Ollama LLM

    Args:
        prompt (str): User's question
        data_summary (str): Summary of the data

    Returns:
        str: Response from Ollama
    """
    try:
        # This is a placeholder - in a real implementation, you would use the Ollama API
        # Example: import ollama; response = ollama.generate(model='qwen3:30b', prompt=prompt)
        ollama_host = os.getenv('OLLAMA_HOST', '192.168.0.103:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen3:30b')

        url = f"http://{ollama_host}/api/generate"

        # Prepare the prompt
        full_prompt = f"データの概要:\n{data_summary}\n\n質問:\n{prompt}\n\n回答:"

        payload = {
            "model": ollama_model,
            "prompt": full_prompt,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return result["response"]

    except Exception as e:
        logger.error(f"Ollama API呼び出しエラー: {str(e)}")
        return f"Ollamaエラー: {str(e)}"

def get_claude_response(prompt, data_summary):
    """
    Get response from Claude LLM

    Args:
        prompt (str): User's question
        data_summary (str): Summary of the data

    Returns:
        str: Response from Claude
    """
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

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        return result["content"][0]["text"]

    except Exception as e:
        logger.error(f"Claude API呼び出しエラー: {str(e)}")
        return f"Claudeエラー: {str(e)}"

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