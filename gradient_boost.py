"""
Script for training a Gradient Boosted Regressor model to capture the feature that contributes to 
high CNV num in a sample, which is likely to be false positives. 

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import contextlib
import shap

# Parameters
DELDUP_XLSX = '/dfs8/pub/ddlin/projects/gradient_boosted_regressor/mdf.xlsx'
TARGET_COL = "#inrun_deldup_calls"
FEATURES_TO_REMOVE = ['#deldup1st_calls', 'Yield(Mbases)', 'Clusters', 'pcr_dup_rate', 'Mean Quality Score']
IQR_MULTI = 2

def load_data(filepath):
    """Loads the data from the specified Excel file."""
    return pd.read_excel(filepath)

def plot_model_fit(y_true, y_pred, model_name):
    """Plots the fit of the model."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} Model Fit")
    plt.savefig(f"{model_name}_fit.png")
    plt.clf()

def train_gradient_boost_with_iqr_filter(df, target_col, iqr_filter_col=None, iqr_range=1.5, drop_cols=None):
    """Trains a gradient boosted regressor and returns the model."""
    # Apply IQR filter on selected column if provided
    if iqr_filter_col:
        Q1 = df[iqr_filter_col].quantile(0.25)
        Q3 = df[iqr_filter_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_range * IQR
        upper_bound = Q3 + iqr_range * IQR
        df = df[(df[iqr_filter_col] >= lower_bound) & (df[iqr_filter_col] <= upper_bound)]

    # Drop specified columns
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Split data
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Set up parameters for GridSearchCV
    param_grid = {
        'n_estimators': [400, 600],
        'learning_rate': [0.01],
        'max_depth': [4, 5]
    }
    gradient = GradientBoostingRegressor(random_state=0)
    grid_search = GridSearchCV(estimator=gradient, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best hyperparameters found by grid search:", best_params)

    # Evaluate and print performance metrics
    evaluate_and_print(best_estimator, X_test, y_test)

    return best_estimator, feature_cols

def evaluate_and_print(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"R-squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

def plot_feature_importance(model, feature_cols):
    """Plots the feature importance."""
    feature_importances = model.feature_importances_
    coeff_df = pd.DataFrame(list(zip(feature_cols, feature_importances)), columns=["Feature", "Coefficient"])
    coeff_df = coeff_df.reindex(coeff_df.Coefficient.abs().sort_values(ascending=False).index)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coeff_df, x="Coefficient", y="Feature")
    plt.title("Feature Importances from GradientBoostingRegressor Model")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.savefig("feature_importances.png")
    plt.clf()

def plot_shap_values(model, X_test, feature_cols):
    """Plots the SHAP values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.savefig("shap_plot.png")
    plt.clf()

def save_model(model, filename):
    """Saves the model to a file using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def run():
    start_time = time.time()

    # Load and process data
    df = load_data(DELDUP_XLSX)
    df = df[['Yield(Mbases)', 'Clusters', '% >= Q30 bases', 'Mean Quality Score', 'specificity', 
             'pcr_dup_rate', 'X100', 'mean_coverage', '#deldup1st_calls', '#inrun_deldup_calls',
             '#nolocos', 'read_count', 'cv_mad_iqr', 'pearson']]

    # Train model
    model, feature_cols = train_gradient_boost_with_iqr_filter(df, TARGET_COL, drop_cols=FEATURES_TO_REMOVE, iqr_range=IQR_MULTI)

    # Plot results
    plot_feature_importance(model, feature_cols)
    plot_shap_values(model, df[feature_cols].values, feature_cols)

    # Save model
    save_model(model, "gradient_model.pkl")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run()
