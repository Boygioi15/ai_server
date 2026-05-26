# Define the absolute path to the Conda env on the server
CONDA_ENV = /home/boygioi15/miniconda3
JOB_DIR := ./job/job_88af5466-1bf9-4787-bf8a-b78fccd3c2be
RUN_ID ?=
ARTIFACT_PATH ?= arima_live_model
IMAGE_NAME ?= arima-mlflow-test

start:
	@echo "Installing requirements via Conda's pip..."
	$(CONDA_ENV)/bin/pip install -r ./requirements.txt
	@echo "Starting FastAPI server..."
	$(CONDA_ENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

log-mlflow:
	python -m app.log_arima_mlflow --job-dir $(JOB_DIR) --artifact-path $(ARTIFACT_PATH)

build-mlflow-docker:
	mlflow models build-docker -m runs:/$(RUN_ID)/$(ARTIFACT_PATH) -n $(IMAGE_NAME)

run-mlflow-docker:
	docker run --rm -p 5001:8080 $(IMAGE_NAME)