
# MLflow Docker test for ARIMA

This repo can now package the trained ARIMA model as an MLflow `pyfunc` model.

## What it does
- Loads your existing `model.pkl`
- Loads `history.parquet` saved from train + validation data
- Accepts `live_data` as an array of requests-per-minute
- Appends that live data to in-memory history
- Returns a forecast for the next `horizon` steps

Default `horizon` is `5`.

## Minimal local flow

1. Train the pipeline so the job folder contains:

```text
config.json
model.pkl
history.parquet
metrics.json
```

2. Log the MLflow model:

```bash
python -m app.log_arima_mlflow --job-dir ./job/job_<job_id>
python -m app.log_arima_mlflow --job-dir ./job/job_88af5466-1bf9-4787-bf8a-b78fccd3c2be
```

The script prints:

- `Run ID`
- `Model URI`

3. Build the Docker image:

```bash
mlflow models build-docker -m runs:/<run_id>/arima_live_model -n arima-mlflow-test
```

mlflow models generate-dockerfile -m runs:/fc17f59e0b324ae5a5e886c925a6f7df/arima_live_model -d "./docker_debug_dir"
mlflow models build-docker \
  -m runs:/fc17f59e0b324ae5a5e886c925a6f7df/arima_live_model \
  -n arima-mlflow-test > docker.log 2>&1

4. Run the container:

```bash
docker run --rm -p 5001:8080 arima-mlflow-test
```

5. Call the model:

```bash
curl -X POST http://127.0.0.1:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "live_data": [101.0, 105.0, 99.0, 110.0],
        "horizon": 5
      }
    ]
  }'
```

## Request contract

Input columns:

- `live_data`: array of numeric requests-per-minute
- `horizon`: optional integer
- `update_history`: optional boolean

Response columns:

- `forecast`: array of forecast values
- `history_size`: base history + accepted live points
- `applied_live_points`: how many new points were appended
- `horizon`: effective forecast horizon

## Important limitation

The history update is currently in-memory inside one serving container.

That means:

- It works fine for a single-container smoke test
- It resets if the container restarts
- It is not shared across multiple replicas

For production, keep the live history in an external store and rebuild the inference state from that store on each request or on a scheduled refresh.

## Lookback / order question

Your ARIMA order itself is not a direct problem here because the fitted `pmdarima` model already carries the internal state it needs from training.

What *does* matter is:

- live data must arrive in correct time order
- the model should see every new minute exactly once
- if points are missing or duplicated, forecasts will drift

So the bigger concern is stream consistency, not manually supplying a separate lookback window.
