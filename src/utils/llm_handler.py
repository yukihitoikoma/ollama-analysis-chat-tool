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

        response = requests.post(url, json=payload, timeout=30000)
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
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": full_prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30000)
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


def detect_analysis_request(prompt):
    """ユーザーの質問から分析リクエストを検出する

    Args:
        prompt (str): ユーザーの質問

    Returns:
        dict: 検出された分析タイプと詳細
    """
    analysis_info = {
        "visualization": None,
        "statistical_analysis": None,
        "model_creation": False
    }

    prompt_lower = prompt.lower()

    # グラフ/可視化の検出
    if any(word in prompt_lower for word in ["散布図", "scatter", "scatterplot"]):
        analysis_info["visualization"] = "scatter"
    elif any(word in prompt_lower for word in ["相関", "correlation", "ヒートマップ", "heatmap"]):
        analysis_info["visualization"] = "correlation"
    elif any(word in prompt_lower for word in ["ヒストグラム", "histogram", "分布"]):
        analysis_info["visualization"] = "histogram"
    elif any(word in prompt_lower for word in ["グラフ", "可視化", "図", "chart", "plot", "visualize", "show"]):
        analysis_info["visualization"] = "auto"

    # 統計分析の検出
    if any(word in prompt_lower for word in ["ロジスティック回帰", "logistic regression", "ロジスティック"]):
        analysis_info["statistical_analysis"] = "logistic_regression"
    elif any(word in prompt_lower for word in ["重回帰", "線形回帰", "linear regression", "回帰分析"]):
        analysis_info["statistical_analysis"] = "linear_regression"
    elif any(word in prompt_lower for word in ["関連性", "association", "アソシエーション"]):
        analysis_info["statistical_analysis"] = "association_analysis"
    elif any(word in prompt_lower for word in ["クラスタリング", "クラスター", "clustering"]):
        analysis_info["statistical_analysis"] = "clustering"

    # モデル作成の検出
    if any(word in prompt_lower for word in ["予測モデル", "機械学習", "モデル作成", "model", "prediction", "予測"]):
        analysis_info["model_creation"] = True

    return analysis_info


def format_analysis_response(analysis_results, llm_response):
    """分析結果とLLM応答を統合してフォーマットする

    Args:
        analysis_results (dict): 分析結果
        llm_response (str): LLMの応答

    Returns:
        str: フォーマットされた応答
    """
    response = llm_response + "\n\n"

    if analysis_results.get("statistical_analysis"):
        response += "---\n\n"
        response += "### 統計分析結果\n\n"
        response += analysis_results["statistical_analysis"]

    if analysis_results.get("model_info"):
        response += "\n\n---\n\n"
        response += "### モデル情報\n\n"
        response += analysis_results["model_info"]

    return response


def get_analysis_interpretation(prompt, data_summary, analysis_text, model_type="ollama"):
    """分析結果の解釈と提案をLLMから取得する

    Args:
        prompt (str): ユーザーの質問
        data_summary (str): データサマリー
        analysis_text (str): 分析結果テキスト
        model_type (str): 使用するLLMのタイプ

    Returns:
        str: LLMによる解釈と提案
    """
    interpretation_prompt = f"""以下のデータ分析結果について、専門家の視点から以下の点をまとめてください:

1. **主要なポイント**: 分析結果から読み取れる重要な発見や傾向
2. **結論**: データから導き出せる結論
3. **次のステップの提案**: さらなる分析や検証のための仮説や提案

データの概要:
{data_summary}

ユーザーの質問:
{prompt}

分析結果:
{analysis_text}

上記を踏まえて、分かりやすく簡潔に日本語で説明してください。"""

    try:
        if model_type == "ollama":
            return get_ollama_interpretation(interpretation_prompt)
        elif model_type == "claude":
            return get_claude_interpretation(interpretation_prompt)
        else:
            return ""
    except Exception as e:
        logger.error(f"解釈取得エラー: {str(e)}")
        return ""


def get_ollama_interpretation(prompt):
    """Ollamaから解釈を取得"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', '192.168.0.103:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen3:30b')
        url = f"http://{ollama_host}/api/generate"

        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=30000)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except Exception as e:
        logger.error(f"Ollama解釈取得エラー: {str(e)}")
        return ""


def get_claude_interpretation(prompt):
    """Claudeから解釈を取得"""
    try:
        claude_api_key = os.getenv("CLAUDE_API")
        claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4")
        if not claude_api_key:
            return ""

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": claude_model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30000)
        response.raise_for_status()
        result = response.json()
        return result["content"][0]["text"]
    except Exception as e:
        logger.error(f"Claude解釈取得エラー: {str(e)}")
        return ""