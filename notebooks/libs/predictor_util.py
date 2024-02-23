import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTE


def make_classification(df: pd.DataFrame, features:list[str], target:str, optim:bool=False) -> list[xgb.XGBClassifier, pd.Series, np.ndarray, object]:    
    X = df[features]
    y = df[target]
    # Splitting the dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if optim:
    # Model initialization
        model = xgb.XGBClassifier(enable_categorical=True, device="cuda")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        # Define the grid of parameters to search
        param_grid = {
            'n_estimators': [50, 100, 205],
            'max_depth': [3, 8, 12],
            'learning_rate': [0.01, 0.05, 0.5],
            # 'reg_lambda': [0, 1],
            # 'reg_alpha': [20, 40, 100],
            'eta': [0.01, 0.08, 0.2],
            'subsample': [0.5, 0.8,1],
            'colsample_bytree': [0.5, 0.8,1],
            'seed': [12, 44, 80],
            'objective': ['multi:softmax']
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
        # Predictions
        y_pred = best_model.predict(X_test)
        print('Best parameters ============', grid_search.best_params_)
        model = best_model

    else:    
        oversampler = RandomOverSampler(sampling_strategy='not majority')
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        # X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
        model = xgb.XGBClassifier(
            enable_categorical=True, 
            n_estimators=150,
            learning_rate=0.08,
            max_depth=7
        )
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)


    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division='warn')

    print("Accuracy:", accuracy)

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