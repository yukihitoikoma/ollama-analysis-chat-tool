import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging



# Set up logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



class DataAnalyzer:

    def __init__(self, df):

        self.df = df



    def get_basic_info(self):

        """基本情報を取得"""

        info = {

            'shape': self.df.shape,

            'columns': list(self.df.columns),

            'dtypes': self.df.dtypes.to_dict(),

            'missing_values': self.df.isnull().sum().to_dict(),

            'memory_usage': self.df.memory_usage(deep=True).sum()

        }

        return info



    def get_descriptive_statistics(self):

        """記述統計量を取得"""

        try:

            # 数値型の列のみを取得

            numeric_df = self.df.select_dtypes(include=[np.number])

            if numeric_df.empty:

                return "数値型の列がありません"

            # 基本的な統計量を計算

            stats = numeric_df.describe()

            return stats

        except Exception as e:

            logger.error(f"記述統計量の計算エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_correlations(self):

        """相関係数を取得"""

        try:

            # 数値型の列のみを取得

            numeric_df = self.df.select_dtypes(include=[np.number])

            if numeric_df.empty:

                return "数値型の列がありません"

            # 相関行列を計算

            correlations = numeric_df.corr()

            return correlations

        except Exception as e:

            logger.error(f"相関係数の計算エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_outliers(self):

        """外れ値を検出"""

        try:

            # 数値型の列のみを取得

            numeric_df = self.df.select_dtypes(include=[np.number])

            if numeric_df.empty:

                return "数値型の列がありません"

            outliers = {}

            for column in numeric_df.columns:

                Q1 = numeric_df[column].quantile(0.25)

                Q3 = numeric_df[column].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR

                upper_bound = Q3 + 1.5 * IQR

                outlier_count = len(numeric_df[(numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)])

                outliers[column] = outlier_count

            return outliers

        except Exception as e:

            logger.error(f"外れ値検出エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_data_types(self):

        """データ型情報を取得"""

        try:

            dtypes_info = self.df.dtypes.value_counts()

            return dtypes_info

        except Exception as e:

            logger.error(f"データ型情報取得エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_missing_data_info(self):

        """欠損値情報を取得"""

        try:

            missing_info = self.df.isnull().sum()

            missing_percent = (missing_info / len(self.df)) * 100

            missing_df = pd.DataFrame({

                '欠損数': missing_info,

                '欠損率(%)': missing_percent

            })

            missing_df = missing_df[missing_df['欠損数'] > 0]

            return missing_df

        except Exception as e:

            logger.error(f"欠損値情報取得エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_categorical_summary(self):

        """カテゴリ型データの要約"""

        try:

            # カテゴリ型の列を取得

            categorical_df = self.df.select_dtypes(include=['object'])

            if categorical_df.empty:

                return "カテゴリ型の列がありません"

            summary = {}

            for column in categorical_df.columns:

                summary[column] = {

                    'unique_count': categorical_df[column].nunique(),

                    'value_counts': categorical_df[column].value_counts().to_dict()

                }

            return summary

        except Exception as e:

            logger.error(f"カテゴリ型データ要約エラー: {str(e)}")

            return f"エラー: {str(e)}"



    def get_descriptive_stats(self):

        """記述統計を取得"""

        return self.df.describe()



    def get_correlation_matrix(self):

        """相関行列を取得"""

        numeric_df = self.df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 1:

            return numeric_df.corr()

        return None



    def get_distribution_plots(self):

        """分布図を取得"""

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        plots = []



        for col in numeric_cols:

            fig = px.histogram(self.df, x=col, title=f"{col}の分布")

            plots.append(fig)



        return plots



    def get_outliers(self):

        """外れ値を検出"""

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        outliers = {}



        for col in numeric_cols:

            Q1 = self.df[col].quantile(0.25)

            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR

            upper_bound = Q3 + 1.5 * IQR



            outliers[col] = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]



        return outliers



    def get_grouped_stats(self, group_col, value_col):

        """グループごとの統計情報を取得"""

        if group_col in self.df.columns and value_col in self.df.columns:

            return self.df.groupby(group_col)[value_col].agg(['mean', 'median', 'std', 'count'])

        return None