import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix

def plot_boxplots(df:pd.DataFrame, columns:list[str], rows:int = 3, cols:int =4) -> None:
  # create a figure with his axes
  fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

  for i, ax in enumerate(axes.flat):
    if i < len(columns):
      # Draw the boxplot for the corresponding column
      df.boxplot(column=[columns[i]], ax=ax)
      # Add a title
      ax.set_title(columns[i])

  plt.subplots_adjust(hspace=0.5)
  plt.tight_layout()
  plt.show()

def analysing_feature_importance(feature_imp_serie: pd.Series):
    # Get feature importances
    series_data = feature_imp_serie
    # Convert the Series to a DataFrame
    feature_importance_df = pd.DataFrame(series_data, columns=['feature_importance'])

    # Reset index to make feature names a column
    feature_importance_df.reset_index(inplace=True)
    feature_importance_df.columns = ['Feature', 'Importance']
    feature_importance_df
    # Plot feature importances
    fig = px.bar(feature_importance_df, x='Feature', y='Importance',
                title='Feature Importances',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                width=1200, height=700)
    fig.show()


def plot_confusion_matrix(data:tuple[pd.Series, np.ndarray]) -> None:
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(data[0], data[1])
    # Plot confusion matrix
    fig = px.imshow(conf_matrix,
        labels=dict(x="Predicted", y="Actual"),
        x=[f"Predicted {i}" for i in range(1, conf_matrix.shape[1] + 1)],
        y=[f"Actual {i}" for i in range(1, conf_matrix.shape[0] + 1)],
        title="Confusion Matrix",
        width=1200, 
        height=800)
    fig.show()