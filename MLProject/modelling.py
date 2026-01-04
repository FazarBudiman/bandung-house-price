import mlflow
import warnings
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "bandung-house-price-preprocessing"
    X_train = pd.read_csv(DATA_DIR / "X_train_processed.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train_processed.csv").values.ravel()

    X_test = pd.read_csv(DATA_DIR / "X_test_processed.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test_processed.csv").values.ravel()

    input_example = X_train[0:5]

    mlflow.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run():    
        model = XGBRegressor(
        n_estimators=466,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.93,
        colsample_bytree=0.6,
        reg_alpha=0.68,
        reg_lambda=1.41,
        random_state=42
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        y_pred = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae )
        mlflow.log_metric("r2", r2)