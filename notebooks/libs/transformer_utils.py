import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def put_a_flag_by_condition(target_col:str, values_list:list[str], df=pd.DataFrame()) -> pd.DataFrame:
  df[f"{target_col}_ISTRUE"] = 0
  for value in values_list:
    df.loc[df[target_col].str.contains(value, case=False, regex=True), f"{target_col}_ISTRUE"] = 1,
  return df

def update_cel_val_by_pattern_filter(target_col:str, values_list:list[str], new_value:str, df=pd.DataFrame) -> pd.DataFrame:
    for value in values_list:
      df.loc[df[target_col].str.contains(value, case=False, regex=True), target_col] = new_value
    return df

def process_features_standardisation(df:pd.DataFrame, 
                                     columns:list[str], 
                                     transformer:BaseEstimator
                                    ) -> pd.DataFrame:
    """
      Dans cette méthode on peut choisire un transformer pour la standardisation parmis:
        - StandardScaler
        - MinMaxScaler
        - MaxAbsScaler 
    """
    preprocessor:BaseEstimator = ColumnTransformer(
      [
        ("scaler", transformer(), columns),
      ],
      verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    new_df:pd.DataFrame = preprocessor.fit_transform(df) # type: ignore
    return new_df

def encode_labels(df:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    le = LabelEncoder()
    for i in cols:
        df[i] = le.fit_transform(df[i])
    return df
  
def one_hot_encode_field(df:pd.DataFrame, field_name:str) -> pd.DataFrame:
      df:pd.DataFrame = df.copy()
      encoder = OneHotEncoder(sparse_output=False, drop="first")
      encoded_data = encoder.fit_transform(df[[field_name]])
      encoded_df = pd.DataFrame(
          encoded_data, 
          columns=encoder.get_feature_names_out([field_name])
      )
      df = pd.concat([df, encoded_df], axis=1)
      df = df.drop(columns=[field_name])
      return df

def remove_outliers(series, threshold=3):
    """
    Remove outliers from a series using Z-score method.
    
    Parameters:
    series (pd.Series): The input series containing numerical data.
    threshold (float): The Z-score threshold. Values beyond this threshold are considered outliers.
    
    Returns:
    pd.Series: The series with outliers removed.
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return series[(z_scores < threshold)]