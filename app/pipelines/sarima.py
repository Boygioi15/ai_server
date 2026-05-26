import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
import joblib
from pmdarima import auto_arima

from contextlib import redirect_stdout
import app.pipelines.common as common
from app.pipelines import LoggerWriter


def sarima_pipeline(job_path):
    common.log_status(job_path, "Pipeline started")
    job_path = Path(job_path)

    # =========================
    # 1. Load config
    # =========================
    with open(job_path / "config.json", "r") as f:
        config = json.load(f)

    chosen_feat = config["feature"]

    # =========================
    # 2. Gate check
    # =========================
    if len(chosen_feat) != 1:
        raise ValueError(
            "ARIMA only supports exactly 1 feature."
        )

    target_col = chosen_feat[0]

    # =========================
    # 3. Load parquet files
    # =========================
    print("Job path: ",job_path)
    common.log_status(job_path, "Loading files")
    train_df = pd.read_parquet(job_path / "train.parquet")
    val_df = pd.read_parquet(job_path / "val.parquet")
    test_df = pd.read_parquet(job_path / "test.parquet")

    # =========================
    # 4. Slice features
    # =========================
    train_df = common.slice_feature(train_df, chosen_feat)
    val_df = common.slice_feature(val_df, chosen_feat)
    test_df = common.slice_feature(test_df, chosen_feat)

    # =========================
    # 5. Convert to series
    # =========================
    train_series = train_df[target_col]
    val_series = val_df[target_col]
    test_series = test_df[target_col]

    # =========================
    # 6. Auto ARIMA
    # =========================
    common.log_status(job_path, "Starting auto_arima search")

    log_file = job_path / "job_status.log"

    logger_writer = LoggerWriter.LoggerWriter(log_file)

    with redirect_stdout(logger_writer):

        model = auto_arima(
            train_series,

            seasonal=True,

            m=1440,

            trace=True,

            error_action="ignore",
            suppress_warnings=True,

            stepwise=True
        )

    # Best discovered order
    best_order = model.order

    print("Best ARIMA order:", best_order)

    # =========================
    # 7. Validation forecast
    # =========================
    common.log_status(job_path, "Generating validation forecast")
    val_pred = model.predict(
        n_periods=len(val_series)
    )

    # =========================
    # 8. Refit on train + val with the best order param from step 6
    # =========================
    combined_series = pd.concat([
        train_series,
        val_series
    ])

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

    # =========================
    # 9. Test forecast
    # =========================
    test_pred = final_model.predict(
        n_periods=len(test_series)
    )

    # =========================
    # 10. Metrics
    # =========================
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

    # =========================
    # 11. Save predictions
    # =========================
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

    # =========================
    # 12. Save metrics
    # =========================
    results = {
        "best_order": best_order,
        "val": val_metrics,
        "test": test_metrics
    }

    with open(job_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # =========================
    # 13. Dump model
    # =========================
    joblib.dump(
        final_model,
        job_path / "model.pkl"
    )
    return {
        "status": "success",
        "results": results
    }