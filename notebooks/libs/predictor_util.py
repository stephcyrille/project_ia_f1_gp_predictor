import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
# from imblearn.over_sampling import RandomOverSampler, SMOTE   
    

def modelfit(df: pd.DataFrame, features:list[str], target:str, model:xgb.XGBClassifier, class_weights:np.ndarray=[]) -> list[object]: 
    X = df[features]
    y = df[target]
    # Splitting the dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=48)
 
    # oversampler = RandomOverSampler(sampling_strategy='not majority')
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    # model.fit(X_train_resampled, y_train_resampled)
    if len(class_weights) > 0:
        model.fit(X_train, y_train, sample_weight=class_weights[y_train])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dtrain_predprob = model.predict_proba(X_train)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division='warn', labels=np.unique(y_pred))

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy)
    print("AUC Score (Train): %f" % roc_auc_score(y_train, dtrain_predprob, multi_class='ovr'))
    
    # Predict on testing data:
    dtest_predprob = model.predict_proba(X_test)
    print('AUC Score (Test): %f' % roc_auc_score(y_test, dtest_predprob, multi_class='ovr'))
                
    # feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

    results = [model, y_test, y_pred, X_train, report, y_train]
    return results

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


