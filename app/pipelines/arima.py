import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from app.pipelines import LoggerWriter
from pmdarima import auto_arima
import io
from contextlib import redirect_stdout
import joblib

import app.pipelines.common as common


def arima_pipeline(job_path):
    common.log_status(job_path, "Pipeline started")
    job_path = Path(job_path)

    # =========================
    # 1. Load config
    # =========================
    common.log_status(job_path, "Step 1: Loading configuration")
    with open(job_path / "config.json", "r") as f:
        config = json.load(f)

    chosen_feat = config["feature"]
    common.log_status(job_path, f"Configuration loaded - Target features: {chosen_feat}")

    # =========================
    # 2. Gate check
    # =========================
    common.log_status(job_path, "Step 2: Validating feature configuration")
    if len(chosen_feat) != 1:
        common.log_status(job_path, f"ERROR: ARIMA requires exactly 1 feature, got {len(chosen_feat)}")
        raise ValueError(
            "ARIMA only supports exactly 1 feature."
        )

    target_col = chosen_feat[0]
    common.log_status(job_path, f"Feature validation passed - Target column: {target_col}")

    # =========================
    # 3. Load parquet files
    # =========================
    print("Job path: ",job_path)
    common.log_status(job_path, "Step 3: Loading parquet datasets")
    train_df = pd.read_parquet(job_path / "train.parquet")
    val_df = pd.read_parquet(job_path / "val.parquet")
    test_df = pd.read_parquet(job_path / "test.parquet")
    common.log_status(job_path, f"Parquet files loaded - Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    # =========================
    # 4. Slice features
    # =========================
    common.log_status(job_path, "Step 4: Slicing feature columns")
    train_df = common.slice_feature(train_df, chosen_feat)
    val_df = common.slice_feature(val_df, chosen_feat)
    test_df = common.slice_feature(test_df, chosen_feat)
    common.log_status(job_path, "Feature slicing completed")

    # =========================
    # 5. Convert to series
    # =========================
    common.log_status(job_path, "Step 5: Converting DataFrames to pandas Series")
    train_series = train_df[target_col]
    val_series = val_df[target_col]
    test_series = test_df[target_col]
    common.log_status(job_path, "Conversion to Series completed")

    # =========================
    # 6. Auto ARIMA
    # =========================
    common.log_status(job_path, "Step 6: Running auto_arima hyperparameter search")

    log_file = job_path / "job_status.log"

    logger_writer = LoggerWriter.LoggerWriter(log_file)

    with redirect_stdout(logger_writer):

        model = auto_arima(
            train_series,

            seasonal=False,

            trace=True,

            error_action="ignore",
            suppress_warnings=True,

            stepwise=True
        )

    # Best discovered order
    best_order = model.order

    print("Best ARIMA order:", best_order)
    common.log_status(job_path, f"Auto ARIMA search completed - Best order: {best_order}")

    # =========================
    # 7. Validation forecast
    # =========================
    common.log_status(job_path, "Step 7: Generating validation set predictions")
    val_pred = model.predict(
        n_periods=len(val_series)
    )
    common.log_status(job_path, f"Validation predictions generated - {len(val_pred)} predictions")

    # =========================
    # 8. Refit on train + val with the best order param from step 6
    # =========================
    common.log_status(job_path, "Step 8: Refitting model on combined train+val data with best parameters")
    combined_series = pd.concat([
        train_series,
        val_series
    ])

    combined_history = pd.DataFrame({
        target_col: combined_series.values
    })

    final_model = auto_arima(
        combined_series,

        seasonal=False,

        start_p=best_order[0],
        d=best_order[1],
        start_q=best_order[2],

        max_p=best_order[0],
        max_q=best_order[2],

        trace=False,

        error_action="ignore",
        suppress_warnings=True,

        stepwise=True
    )
    common.log_status(job_path, f"Model refit completed on {len(combined_series)} combined samples")

    # =========================
    # 9. Test forecast
    # =========================
    common.log_status(job_path, "Step 9: Generating test set predictions")
    test_pred = final_model.predict(
        n_periods=len(test_series)
    )
    common.log_status(job_path, f"Test predictions generated - {len(test_pred)} predictions")

    # =========================
    # 10. Metrics
    # =========================
    common.log_status(job_path, "Step 10: Computing evaluation metrics")
    val_metrics = {
        "mae": float(
            mean_absolute_error(
                val_series,
                val_pred
            )
        ),
        "rmse": float(
            root_mean_squared_error(
                val_series,
                val_pred,
            )
        ),

        "r2": float(
            r2_score(
                val_series,
                val_pred
            )
        )
    }

    test_metrics = {
        "mae": float(
            mean_absolute_error(
                test_series,
                test_pred
            )
        ),

        "mse": float(
            root_mean_squared_error(
                test_series,
                test_pred
            )
        ),

        "rmse": float(
            root_mean_squared_error(
                test_series,
                test_pred,
            )
        ),

        "r2": float(
            r2_score(
                test_series,
                test_pred
            )
        )
    }
    common.log_status(job_path, f"Metrics computed - Val MAE: {val_metrics['mae']:.4f}, Test RMSE: {test_metrics['rmse']:.4f}")

    # =========================
    # 11. Save predictions
    # =========================
    common.log_status(job_path, "Step 11: Saving prediction results to parquet")
    val_output = pd.DataFrame({
        "actual": val_series.values,
        "prediction": val_pred
    })

    test_output = pd.DataFrame({
        "actual": test_series.values,
        "prediction": test_pred
    })

    val_output.to_parquet(
        job_path / "val_predictions.parquet"
    )

    test_output.to_parquet(
        job_path / "test_predictions.parquet"
    )
    common.log_status(job_path, "Prediction files saved successfully")

    # =========================
    # 11b. Save combined history for deployment/inference warm start
    # =========================
    common.log_status(job_path, "Step 11b: Saving combined training history for inference")
    combined_history.to_parquet(
        job_path / "history.parquet"
    )
    common.log_status(job_path, "Combined history saved successfully")

    # =========================
    # 12. Save metrics
    # =========================
    common.log_status(job_path, "Step 12: Saving metrics to JSON")
    results = {
        "best_order": best_order,
        "val": val_metrics,
        "test": test_metrics
    }

    with open(job_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    common.log_status(job_path, "Metrics file saved successfully")

    # =========================
    # 13. Dump model
    # =========================
    common.log_status(job_path, "Step 13: Saving trained model to pickle file")
    joblib.dump(
        final_model,
        job_path / "model.pkl"
    )
    common.log_status(job_path, "Model file saved successfully")
    common.log_status(job_path, "Pipeline execution completed successfully")
    return {
        "status": "success",
        "results": results
    }
