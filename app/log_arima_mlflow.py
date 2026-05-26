from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from app.mlflow_arima_model import ArimaLiveForecastModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log an ARIMA pipeline as an MLflow pyfunc model.")
    parser.add_argument("--job-dir", required=True, help="Path to the training job directory.")
    parser.add_argument("--artifact-path", default="arima_live_model", help="MLflow artifact path.")
    parser.add_argument("--registered-model-name", default=None, help="Optional MLflow registered model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir).resolve()

    model_path = job_dir / "model.pkl"
    history_path = job_dir / "history.parquet"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history artifact: {history_path}")

    input_example = pd.DataFrame([
        {
            "live_data": [120.0, 125.0, 121.0],
            "horizon": 5,
        }
    ])
    output_example = pd.DataFrame([
        {
            "forecast": [0.0] * 5,
            "history_size": 0,
            "applied_live_points": 3,
            "horizon": 5,
        }
    ])

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path,
            python_model=ArimaLiveForecastModel(),
            code_paths=["app"],
            artifacts={
                "model": str(model_path),
                "history": str(history_path),
            },
            pip_requirements=[
                "mlflow",
                "pandas",
                "joblib",
                "pmdarima",
                "pyarrow",
            ],
            input_example=input_example,
            signature=infer_signature(input_example, output_example),
            registered_model_name=args.registered_model_name,
        )

        print("Logged MLflow model successfully.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact path: {args.artifact_path}")
        print(f"Model URI: runs:/{run.info.run_id}/{args.artifact_path}")


if __name__ == "__main__":
    main()
