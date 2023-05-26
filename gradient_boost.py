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
from pathlib import Path
import shutil
import shap

# deldup df with uniformity info
deldup_xlsx = '/home/tealeave/projects/CNVqc/mdf.xlsx'
# run_xlsx = '/mnt/dlin/tickets/BioHD-359_High_Deldup_Invest/deldup_study_scikit/PF_gmdf.xlsx'

def plot_model_fit(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} Model Fit")
    # Save the plot as a PNG file
    plt.savefig(f"{model_name}_fit.png")
    plt.clf()

def train_gradient_boost_with_iqr_filter(df, target_col, iqr_filter_col=None, iqr_range=1.5, drop_cols=None):
    # Apply IQR filter on selected column if provided
    if iqr_filter_col is not None:
        Q1 = df[iqr_filter_col].quantile(0.25)
        Q3 = df[iqr_filter_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_range * IQR
        upper_bound = Q3 + iqr_range * IQR
        df = df[(df[iqr_filter_col] >= lower_bound) & (df[iqr_filter_col] <= upper_bound)]

    # Drop the selected columns if provided
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    # Prepare data
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # # If using gridsearch
    # # Define the parameter grid for hyperparameter tuning
    # param_grid = {
    #     'n_estimators': [ 400, 600],
    #     'learning_rate': [ 0.01],
    #     'max_depth': [ 4, 5],
    # }

    # # Initialize the GradientBoostingRegressor
    # gradient = GradientBoostingRegressor(random_state=0)

    # # Perform grid search with cross-validation
    # grid_search = GridSearchCV(estimator=gradient, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    # grid_search.fit(X_train, y_train)

    # # Get the best estimator
    # best_estimator = grid_search.best_estimator_

    # # print the best hyperparameters
    # # Best hyperparameters found by grid search:
    # # {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 400}
    # print("Best hyperparameters found by grid search:")
    # best_params = grid_search.best_params_
    # print(best_params)



    # without using gridsearch
    # Replace the parameter grid with your desired parameters
    best_params = {
        'n_estimators': 400,
        'learning_rate': 0.01,
        'max_depth': 5,
        'verbose':1
    }
    
    # Initialize the GradientBoostingRegressor with the best parameters
    best_estimator = GradientBoostingRegressor(random_state=0, **best_params)

    # Fit the model to the training data
    best_estimator.fit(X_train, y_train)



    # Evaluate the model performance
    y_pred = best_estimator.predict(X_test)
    score = best_estimator.score(X_test, y_test)

    # Print the feature importances
    feature_importances = best_estimator.feature_importances_
    for col, importance in zip(feature_cols, feature_importances):
        print(f"{col}: {importance}")

    # Plot the feature importances
    coeff_df = pd.DataFrame(list(zip(feature_cols, feature_importances)), columns=["Feature", "Coefficient"])
    coeff_df = coeff_df.reindex(coeff_df.Coefficient.abs().sort_values(ascending=False).index)
    custom_palette = coeff_df["Coefficient"].apply(lambda x: "blue" if x >= 0 else "red").tolist()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coeff_df, x="Coefficient", y="Feature", palette=custom_palette)
    plt.title("Feature Importances from GradientBoostingRegressor Model")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    # Save the plot as a PNG file
    plt.savefig("feature_importances.png")
    plt.clf()
    
    # Calculate and print the quantitative metrics
    # R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print("\nHyperparameters used:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    y_pred = best_estimator.predict(X_test)  # Make sure to get the predictions from the test set
    plot_model_fit(y_test, y_pred, "GradientBoostingRegressor")
    
    # SHAP
    # Calculate SHAP values
    explainer = shap.TreeExplainer(best_estimator)
    shap_values = explainer.shap_values(X_test)

    # Save the SHAP values for later use
    with open("shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    # Plot SHAP values
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.savefig("shap_plot.png")
    plt.clf()

    return best_estimator, best_params

def run():
    start_time = time.time()

    # Load the deldup data
    sample_df = pd.read_excel(deldup_xlsx)
    sample_corr_df = sample_df[['Yield(Mbases)', 'Clusters', '% >= Q30 bases', 'Mean Quality Score',
       'specificity', 'pcr_dup_rate', 'X100', 'mean_coverage', '#deldup1st_calls', '#inrun_deldup_calls',
       '#nolocos', 'read_count', 'cv_mad_iqr', 'pearson']]

    # run_df = sample_df.groupby('runid').mean()
    # run_corr_df = run_df[['Yield(Mbases)', 'Clusters', '% >= Q30 bases', 'Mean Quality Score',
    # 'specificity', 'pcr_dup_rate', 'mean_coverage', '#deldup1st_calls', '#inrun_deldup_calls',
    # '#nolocos', 'read_count', 'cv_mad_iqr', 'pearson']]

    # run_df = pd.read_excel(run_xlsx)
    # print(run_df.columns)


    # Set parameters
    target_col = "#inrun_deldup_calls"
    selected_features_to_remove = ['#deldup1st_calls', 'Yield(Mbases)', 'Clusters', 'pcr_dup_rate', 'Mean Quality Score'
                                   ]
    iqr_multi = 2

    # Run the function and redirect the output to the "performance.txt" file
    with open("performance.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            gradient_model, best_params = train_gradient_boost_with_iqr_filter(sample_corr_df, target_col, iqr_filter_col=None,
                                                    iqr_range=iqr_multi, drop_cols=selected_features_to_remove)
    # Serialize the gradient_model
    with open("gradient_model.pkl", "wb") as f:
        pickle.dump(gradient_model, f)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # Create the subfolder with the hyperparameters in the name
    best_params_str = "_".join([f"{k}={v}" for k, v in best_params.items()])
    subdir = Path(f"model_{best_params_str}")
    subdir.mkdir(parents=True, exist_ok=True)
    # Move the generated files into the subfolder
    for file_name in ["feature_importances.png", "performance.txt", "gradient_model.pkl", "GradientBoostingRegressor_fit.png", "shap_plot.png", 'shap_values.pkl']:
        shutil.move(file_name, subdir / file_name)


    # # Load the serialized gradient_model
    # with open("gradient_model.pkl", "rb") as f:
    #     loaded_gradient_model = pickle.load(f)

    # # Use the loaded_gradient_model for predictions
    # # y_pred = loaded_gradient_model.predict(X_test)


if __name__=="__main__":
    run()