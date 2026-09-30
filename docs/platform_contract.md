# AWARE workbench API contract v1

Single-user demonstration and analysis workbench. FastAPI serves `/api` and the built React SPA. Browser fetch uses same-origin HttpOnly session cookies and JSON except file endpoints. In private mode all `/api` endpoints except auth status/setup/login/logout and health require authentication. Logout is idempotent and clears stale cookies. JSON errors: `{"detail":"readable message"}`. Times are UTC ISO8601. Training labels are never inferred from prediction data.

- `GET /api/health` -> `{status:"ok"}`
- `GET /api/auth/status` -> `{configured:boolean,authenticated:boolean,demo_mode:boolean}`
- `POST /api/auth/setup`, `/api/auth/login` JSON `{password:string}` -> `{ok:true}`; setup once; password minimum 12 chars.
- `POST /api/auth/logout` -> `{ok:true}`
- `GET /api/dashboard` -> `{datasets_count,runs_count,completed_runs,prediction_batches,recent_runs:Run[],recent_activity:Activity[]}`
- `GET /api/datasets` -> `{items:Dataset[]}`
- `POST /api/datasets` multipart `file` -> Dataset detail
- `POST /api/datasets/demo` -> a persisted synthetic example Dataset, explicitly marked simulated.
- `GET /api/datasets/{id}` -> Dataset detail
- `GET /api/datasets/{id}/download` -> original CSV
- `GET /api/runs` -> `{items:Run[]}`
- `POST /api/runs` -> Run (queued); JSON `{dataset_id,name,target_column,positive_value,positive_label,id_column:null|string,feature_columns:string[],categorical_columns:string[],models:string[],seed:42}`
- `GET /api/runs/{id}` -> Run including result when completed. Poll while queued/running.
- `GET /api/runs/{id}/report` -> downloadable JSON result with run config and provenance.
- `GET /api/runs/{id}/template` -> CSV with configured ID plus feature headers and no records, for scoring.
- `POST /api/runs/{id}/predict` multipart `file`, `model`, `calibration` (`raw|platt|isotonic`) -> PredictionBatch
- `GET /api/predictions` -> `{items:PredictionBatch[]}`
- `GET /api/predictions/{id}/download` -> scored CSV (record_id, probability, missing_count; no automated business decision).

`Dataset`: `{id,name,rows,column_count,missing_rate,created_at,source:"upload"|"demo",columns:Column[],preview:Record<string,string|null>[],warnings:string[]}`. List may include full columns and preview. `Column`: `{name,kind:"numeric"|"categorical",missing_count,missing_rate,unique_count,sample_values:string[]}`; at most 8 sample values. CSV limit 20 MiB, 50,000 rows, 100 columns, UTF-8/BOM. Empty fields and `?` are missing; category strings NA/None etc remain legitimate unless empty. Any inferred kind can be overridden at run configuration by categorical_columns. Duplicate/empty column names rejected.

`Run`: `{id,name,dataset_id,dataset_name,status:"queued"|"running"|"completed"|"failed",progress:number,message:string,error:null|string,created_at,completed_at:null|string,config:RunConfig,result:null|RunResult}`.

`RunResult`: `{rows:number,positive_label:string,split_sizes:{train,calibration,test},class_counts:{train:number[],calibration:number[],test:number[]},metrics:Metric[],feature_quality:Quality[],warnings:string[],provenance:object}`.
`Metric`: `{model,calibration,auc,brier,ece,f1,log_loss,reliability:{counts:number[],mean_probability:(number|null)[],positive_frequency:(number|null)[]}}`.
`Quality`: `{name,train_missing_rate,test_missing_rate}`.
`Activity`: `{id,action,detail,created_at}`.
`PredictionBatch`: `{id,run_id,run_name,model,calibration,rows,mean_probability,missing_rate,created_at,positive_label,preview:{record_id:string,probability:number,missing_count:number}[],quality:{name,training_missing_rate,batch_missing_rate,unseen_rate:number}[],warnings:string[]}`. Lists use same shape. API never returns model pickle paths, password hashes or session tokens.

Models: `logistic` (Logistic Regression), `random_forest` (Random Forest), `lightgbm` (LightGBM), `mlp` (MLP control), `mask_aware_mlp` (MLP + mask). UI default logistic/random_forest; all selected alternatives and calibrations reported, no automatic selection on the test set.

## Pure engine contract (service/engine.py)

- `read_csv_bytes(content: bytes) -> pandas.DataFrame` (string columns and NaNs; strict limits and headers)
- `profile_frame(frame) -> dict` (`rows,column_count,missing_rate,columns,preview,warnings`)
- `demo_frame() -> DataFrame` realistic-looking **synthetic** credit predictors and 0/1 target `outcome`; at least 500 rows and deliberate missingness.
- `validate_training(frame, config: dict) -> None`, including target/feature/id/cardinality checks; raise ValueError with readable messages.
- `train_analysis(frame, config: dict, progress: Callable[[int,str],None]|None = None) -> (result: dict, bundle: object)`. Independent approximately 60/20/20 grouped stratified holdout, train-only category mapping/imputation, fixed hyperparameters, no test selection. Record split indices/fingerprints, packages/seed/model parameters. All exported JSON finite or null.
- `score_batch(bundle, frame, model: str, calibration: str) -> (summary: dict, scored_frame: DataFrame)`; frozen schema, missing required columns fail, numeric coercion failures fail, unknown categories handled without refit and explicitly reported, extra columns ignored with warning; preserve row order. Summary matches PredictionBatch minus API identity/timestamps/run_name. Score output probability means configured positive event. If an ID column was configured, require nonmissing unique IDs (both training and scoring); otherwise assign stable 1-based row IDs. Never echo raw predictor rows in scoring output.

## Demo-first product scope

The user selected product demonstrations/defense presentations. `AWARE_DEMO_MODE=1` skips account setup only for loopback-bound local service; auth/status includes demo_mode=true and authenticated=true. Startup seeds a clearly labeled synthetic dataset and one genuine LR/RF background analysis when storage is empty. Non-demo mode retains password setup/login. No mocked scores or model rankings. PredictionBatch also includes histogram:{edges:number[11],counts:number[10]} calculated over the full scored batch. Scoring templates include configured ID plus feature names.
