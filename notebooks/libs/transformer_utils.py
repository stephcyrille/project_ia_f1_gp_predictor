import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


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