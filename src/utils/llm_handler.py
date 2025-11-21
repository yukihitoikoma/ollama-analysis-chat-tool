import os
import logging
from dotenv import load_dotenv
import requests
import json
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_external_image_urls(text):
    """
    Remove external image URLs from text

    Args:
        text (str): Text containing potential image URLs

    Returns:
        str: Text with external image URLs removed
    """
    # Remove markdown image links with external URLs
    # Pattern: ![alt text](http://... or https://...)
    text = re.sub(r'!\[([^\]]*)\]\(https?://[^\)]+\)', '', text)

    # Remove plain external image URLs
    text = re.sub(r'https?://[^\s]+\.(png|jpg|jpeg|gif|svg)', '', text, flags=re.IGNORECASE)

    # Clean up extra blank lines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    return text.strip()

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
            response = get_ollama_response(prompt, data_summary)
        elif model_type == "claude":
            response = get_claude_response(prompt, data_summary)
        else:
            return "Unknown model type"

        # 外部画像URLを削除
        response = remove_external_image_urls(response)
        return response
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
        if not ollama_host.startswith(('http://', 'https://')):
            # スキームがない場合はhttpを追加
            ollama_url = f"http://{ollama_host}"
        else:
            # スキームがある場合はそのまま使用
            ollama_url = ollama_host
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen3:30b')

        url = f"{ollama_url}/api/generate"

        # Prepare the prompt
        full_prompt = f"""データの概要:
{data_summary}

質問:
{prompt}

注意:
- グラフや画像のURLリンク（imgur.comなど）を生成しないでください
- グラフはシステムが自動的に生成します
- テキストでの説明のみを提供してください

回答:"""

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
        full_prompt = f"""データの概要:
{data_summary}

質問:
{prompt}

注意:
- グラフや画像のURLリンク（imgur.comなど）を生成しないでください
- グラフはシステムが自動的に生成します
- テキストでの説明のみを提供してください

回答:"""

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
        "statistical_analysis": None,
        "model_creation": False
    }

    prompt_lower = prompt.lower()

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

注意:
- グラフや画像のURLリンク（imgur.comなど）を生成しないでください
- グラフはシステムが自動的に生成します
- テキストでの説明のみを提供してください

上記を踏まえて、分かりやすく簡潔に日本語で説明してください。"""

    try:
        if model_type == "ollama":
            response = get_ollama_interpretation(interpretation_prompt)
        elif model_type == "claude":
            response = get_claude_interpretation(interpretation_prompt)
        else:
            return ""

        # 外部画像URLを削除
        response = remove_external_image_urls(response)
        return response
    except Exception as e:
        logger.error(f"解釈取得エラー: {str(e)}")
        return ""


def get_ollama_interpretation(prompt):
    """Ollamaから解釈を取得"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', '192.168.0.103:11434')
        if not ollama_host.startswith(('http://', 'https://')):
            # スキームがない場合はhttpを追加
            ollama_url = f"http://{ollama_host}"
        else:
            # スキームがある場合はそのまま使用
            ollama_url = ollama_host
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen3:30b')
        url = f"{ollama_url}/api/generate"

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


def get_required_graphs(prompt, data_summary, response_text, model_type="ollama", already_planned_graphs=None):
    """LLMの応答から必要なグラフのリストを取得する

    Args:
        prompt (str): ユーザーの質問
        data_summary (str): データサマリー
        response_text (str): LLMの応答テキスト
        model_type (str): 使用するLLMのタイプ
        already_planned_graphs (list): 既に生成予定のグラフタイプのリスト

    Returns:
        list: 必要なグラフのタイプリスト（例: ["correlation", "histogram"]）、不要な場合は空リスト
    """
    if already_planned_graphs is None:
        already_planned_graphs = []
    # 既に計画されているグラフがある場合は除外情報を追加
    exclusion_text = ""
    if already_planned_graphs:
        exclusion_text = f"\n\n注意: 以下のグラフは既に生成予定です。これらを除外してください:\n- {', '.join(already_planned_graphs)}"

    graph_prompt = f"""以下のユーザーの質問とその回答を分析し、グラフが必要かどうかを判断してください。

ユーザーの質問:
{prompt}

回答:
{response_text}

指示:
1. 回答の内容から、データ分析やグラフ表示が必要かどうかを判断してください
2. 単なる質問応答やテキストのみで十分な場合は「none」と返してください
3. グラフが必要な場合のみ、以下から適切なタイプを選んでカンマ区切りで返してください

利用可能なグラフタイプ:
- correlation: 相関ヒートマップ（変数間の関係を見る場合）
- histogram: ヒストグラム（データの分布を見る場合）
- scatter: 散布図（2変数の関係を見る場合）
- bar: 棒グラフ（カテゴリ別の比較）
- box: 箱ひげ図（分布の統計的比較）{exclusion_text}

重要: 回答内容が数値分析や統計、データの傾向に言及していない場合は「none」と返してください。

必要なグラフタイプのみを返してください（説明文は不要）:"""

    try:
        if model_type == "ollama":
            response = get_ollama_response(graph_prompt, "")
        elif model_type == "claude":
            response = get_claude_response(graph_prompt, "")
        else:
            return []

        # 応答からグラフタイプを抽出
        response = response.strip().lower()

        # "none"または"不要"が含まれている場合は空リストを返す
        if "none" in response or "不要" in response or "なし" in response:
            logger.info("グラフは不要と判断されました")
            return []

        # カンマまたは改行で分割
        graph_types = [g.strip() for g in re.split(r'[,\n]', response) if g.strip()]
        # 有効なグラフタイプのみをフィルタ
        valid_types = ["correlation", "histogram", "scatter", "bar", "line", "box", "regression", "feature_importance", "clustering"]
        filtered_types = [g for g in graph_types if g in valid_types]

        # 既に計画されているグラフを除外
        filtered_types = [g for g in filtered_types if g not in already_planned_graphs]

        # 重複を削除
        filtered_types = list(dict.fromkeys(filtered_types))

        logger.info(f"要求されたグラフタイプ（除外後）: {filtered_types if filtered_types else '不要'}")
        return filtered_types

    except Exception as e:
        logger.error(f"グラフタイプ取得エラー: {str(e)}")
        return []  # エラー時は空リスト