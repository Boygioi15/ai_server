import json
import shutil
import uuid
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks
)

from app.pipelines.arima import arima_pipeline
from app.pipelines.sarima import sarima_pipeline

app = FastAPI()

# =========================
# Base job directory
# =========================
BASE_DIR = Path(__file__).resolve().parent

JOB_PATH = BASE_DIR.parent / "job"

JOB_PATH.mkdir(
    exist_ok=True
)


# =========================
# Background wrapper
# =========================
def run_pipeline(job_dir: str):

    try:
        arima_pipeline(job_dir)

    except Exception as e:
        print(f"Pipeline failed: {e}")


# =========================
# API Endpoint
# =========================
@app.post("/train_job")
async def train_model(
    background_tasks: BackgroundTasks,

    config: str = Form(...),

    train_file: UploadFile = File(...),
    val_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    try:
        print("Hi")
        # =========================
        # 1. Parse config
        # =========================
        config_data = json.loads(config)

        # =========================
        # 2. Create job folder
        # =========================
        job_id = str(uuid.uuid4())

        job_dir = JOB_PATH / f"job_{job_id}"

        job_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        # =========================
        # 3. Save config.json
        # =========================
        with open(job_dir / "config.json", "w") as f:

            json.dump(
                config_data,
                f,
                indent=4
            )

        # =========================
        # 4. Save uploaded files
        # =========================
        file_mapping = {
            "train.parquet": train_file,
            "val.parquet": val_file,
            "test.parquet": test_file
        }

        for filename, uploaded_file in file_mapping.items():

            save_path = job_dir / filename

            uploaded_file.file.seek(0)

            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(
                    uploaded_file.file,
                    buffer
                )

        # =========================
        # 5. Run pipeline in background
        # =========================
        background_tasks.add_task(
            run_pipeline,
            str(job_dir)
        )

        # =========================
        # 6. Return immediately
        # =========================
        return {
            "status": "accepted",

            "job_id": job_id,

            "job_dir": str(job_dir),

            "config": config_data,

            "saved_files": [
                "train.parquet",
                "val.parquet",
                "test.parquet"
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )