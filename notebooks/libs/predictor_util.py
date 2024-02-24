import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE   
    

def modelfit(df: pd.DataFrame, features:list[str], target:str, model:xgb.XGBClassifier, optim:bool=False) -> list[object]: 
    X = df[features]
    y = df[target]
    # Splitting the dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
    # oversampler = RandomOverSampler(sampling_strategy='not majority')
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    # model.fit(X_train_resampled, y_train_resampled)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dtrain_predprob = model.predict_proba(X_train)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division='warn', labels=np.unique(y_pred))

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy)
    print("AUC Score (Train): %f" % roc_auc_score(y_train, dtrain_predprob, multi_class='ovo'))
    
    # Predict on testing data:
    dtest_predprob = model.predict_proba(X_test)
    print('AUC Score (Test): %f' % roc_auc_score(y_test, dtest_predprob, multi_class='ovo'))
                
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    results = [model, y_test, y_pred, X_train, report]
    return results


def analysing_feature_importance(model:xgb.XGBClassifier, df_train: pd.DataFrame):
    # Get feature importances
    feature_importances = model.feature_importances_
    # Get feature names
    feature_names = df_train.columns
    # Create DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
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

def plot_auc_roc(data:tuple[pd.Series, np.ndarray]) -> None:
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(data[0], data[1])
    roc_auc = auc(fpr, tpr)

    # Create a DataFrame for ROC curve data
    roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})

    # Plot ROC curve
    fig = px.line(roc_df, x='False Positive Rate', y='True Positive Rate',
                title=f'Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.2f})',
                labels={'False Positive Rate': 'False Positive Rate', 'True Positive Rate': 'True Positive Rate'})
    fig.show()


