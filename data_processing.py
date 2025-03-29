import pandas as pd
import numpy as np

class DataProcessing:
    def __init__(self):
        pass

    @staticmethod
    def get_csv_data(path: str, **kwargs) -> pd.DataFrame:
        """
        Gets data from csv file
        :param path: filepath
        :return: DataFrame of the CSV file
        """
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def drop_dataframe_columns(df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """
        Drops a given column
        :param df: dataframe
        :param column: Column name to be dropped
        :return: DataFrame without the specified column
        """
        return df.drop(columns=[column], **kwargs)

    @staticmethod
    def sort_dataframe(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        """
        Sorts a dataframe based on column name and ascending order
        :param df: DataFrame
        :param column: Column name to be sorted
        :param ascending: Boolean
        :return: Sorted dataframe
        """
        return df.sort_values(by=[column], ascending=ascending)

    @staticmethod
    def split_x_y_dataframe(df: pd.DataFrame, x_column: str|list, y_column: str) -> tuple:
        """
        Splits a dataframe into x_column and y_column
        :param df: DataFrame
        :param x_column: Column name
        :param y_column: Target column name
        :return: Tuple of x_column and y_column
        """
        return df[x_column], df[y_column]

