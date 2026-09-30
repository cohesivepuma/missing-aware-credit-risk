"""HTTP workflow and trust-boundary checks for the persisted AWARE workbench."""

import csv
import io
import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service.api import create_app
from service.engine import demo_frame


PASSWORD = "local-workbench-test-password"
MAX_UPLOAD = 20 * 1024 * 1024
UNKNOWN_ID = "0" * 32


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def post_csv(client: TestClient, url: str, content: bytes, **fields):
    return client.post(url, files={"file": ("credit.csv", content, "text/csv")}, data=fields)


def completed_run(client: TestClient, identifier: str) -> dict:
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{identifier}")
        assert response.status_code == 200, response.text
        run = response.json()
        if run["status"] not in {"queued", "running"}:
            assert run["status"] == "completed", run
            return run
        time.sleep(0.04)
    pytest.fail(f"Analysis {identifier} did not finish within 12 seconds")


@pytest.fixture(scope="module")
def trained_source(tmp_path_factory):
    """Fit a real LR once; individual tests receive independent persisted copies."""
    directory = tmp_path_factory.mktemp("platform-training")
    frame = demo_frame().iloc[:600].copy()
    frame.insert(0, "case_id", [f"case-{index:04d}" for index in range(len(frame))])
    content = csv_bytes(frame)
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        response = post_csv(client, "/api/datasets", content)
        assert response.status_code == 201, response.text
        dataset = response.json()
        assert dataset["rows"] == len(frame)
        assert dataset["source"] == "upload"
        features = [name for name in frame if name not in {"case_id", "outcome"}]
        config = {
            "dataset_id": dataset["id"], "name": "Synthetic credit workflow",
            "target_column": "outcome", "positive_value": "1",
            "positive_label": "Simulated overdue event", "id_column": "case_id",
            "feature_columns": features,
            "categorical_columns": [column["name"] for column in dataset["columns"]
                                    if column["kind"] == "categorical" and column["name"] in features],
            "models": ["logistic"], "seed": 42,
        }
        response = client.post("/api/runs", json=config)
        assert response.status_code == 202, response.text
        run = completed_run(client, response.json()["id"])
        assert run["progress"] == 100 and run["completed_at"]
        assert run["config"] == config
        result = run["result"]
        assert result["rows"] == len(frame)
        assert sum(result["split_sizes"].values()) == len(frame)
        assert {(metric["model"], metric["calibration"]) for metric in result["metrics"]} == {
            ("logistic", method) for method in ("raw", "platt", "isotonic")
        }
        for metric in result["metrics"]:
            assert sum(metric["reliability"]["counts"]) == result["split_sizes"]["test"]
        json.dumps(result, allow_nan=False)
    return directory, frame, content, dataset, run


@pytest.fixture
def trained_workspace(tmp_path, trained_source):
    directory, frame, content, dataset, run = trained_source
    shutil.copytree(directory, tmp_path, dirs_exist_ok=True)
    return tmp_path, frame.copy(), content, dataset, run


def test_csv_training_scoring_downloads_and_restart_persistence(trained_workspace):
    directory, frame, original, dataset, run = trained_workspace
    scoring = frame.iloc[:73].drop(columns="outcome").copy()
    scoring.loc[0, "case_id"] = "=SUM(1,1)"
    scoring.loc[0, "annual_income"] = np.nan
    scoring.loc[1, "employment_type"] = "previously-unseen-category"
    scoring["private_notes"] = "private-value-must-not-be-exported"
    features = run["config"]["feature_columns"]
    batches, downloads = [], {}
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        assert client.get("/api/auth/status").json()["demo_mode"] is True
        assert client.get(f"/api/datasets/{dataset['id']}/download").content == original
        template = client.get(f"/api/runs/{run['id']}/template")
        assert template.status_code == 200
        assert list(csv.reader(io.StringIO(template.content.decode("utf-8-sig")))) == [
            ["case_id", *features]
        ]
        report = client.get(f"/api/runs/{run['id']}/report")
        assert report.status_code == 200
        assert "attachment" in report.headers["content-disposition"]
        assert report.json()["config"] == run["config"]
        assert report.json()["result"] == run["result"]
        assert "bundle_sha256" not in report.json()
        assert ".joblib" not in report.text

        for calibration in ("raw", "platt", "isotonic"):
            response = post_csv(client, f"/api/runs/{run['id']}/predict", csv_bytes(scoring),
                                model="logistic", calibration=calibration)
            assert response.status_code == 201, response.text
            batch = response.json()
            batches.append(batch)
            assert batch["rows"] == len(scoring) > len(batch["preview"])
            assert batch["positive_label"] == run["config"]["positive_label"]
            assert all(set(row) == {"record_id", "probability", "missing_count"}
                       for row in batch["preview"])
            assert "private-value-must-not-be-exported" not in response.text
            assert any("private_notes" in warning for warning in batch["warnings"])
            assert next(item for item in batch["quality"] if item["name"] == "employment_type")["unseen_rate"] > 0
            output = client.get(f"/api/predictions/{batch['id']}/download")
            assert output.status_code == 200
            assert "attachment" in output.headers["content-disposition"]
            downloads[batch["id"]] = output.content
            scored = pd.read_csv(io.BytesIO(output.content), dtype={"record_id": str})
            assert scored.columns.tolist() == ["record_id", "probability", "missing_count"]
            # Spreadsheet formula identifiers are escaped; ordinary IDs retain order.
            assert scored["record_id"].tolist() == ["'=SUM(1,1)", *scoring["case_id"].iloc[1:].tolist()]
            assert scored["probability"].between(0, 1).all()
            assert batch["mean_probability"] == pytest.approx(scored["probability"].mean())
            np.testing.assert_array_equal(scored["missing_count"], scoring[features].isna().sum(axis=1))
            assert batch["missing_rate"] == pytest.approx(scoring[features].isna().to_numpy().mean())
            expected, edges = np.histogram(scored["probability"], bins=np.linspace(0, 1, 11))
            assert batch["histogram"]["counts"] == expected.tolist()
            assert batch["histogram"]["edges"] == edges.tolist()
            assert sum(batch["histogram"]["counts"]) == len(scoring)
            json.dumps(batch, allow_nan=False)

        dashboard = client.get("/api/dashboard").json()
        assert (dashboard["datasets_count"], dashboard["runs_count"], dashboard["completed_runs"],
                dashboard["prediction_batches"]) == (1, 1, 1, 3)
        assert {item["action"] for item in dashboard["recent_activity"]} >= {
            "dataset_imported", "analysis_completed", "batch_scored"
        }

    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as restarted:
        assert restarted.get("/api/datasets").json()["items"][0]["id"] == dataset["id"]
        assert restarted.get(f"/api/runs/{run['id']}").json() == run
        persisted = {item["id"]: item for item in restarted.get("/api/predictions").json()["items"]}
        assert persisted == {batch["id"]: batch for batch in batches}
        for identifier, expected in downloads.items():
            assert restarted.get(f"/api/predictions/{identifier}/download").content == expected
        again = post_csv(restarted, f"/api/runs/{run['id']}/predict", csv_bytes(scoring),
                         model="logistic", calibration="raw")
        assert again.status_code == 201, again.text
        assert restarted.get(f"/api/predictions/{again.json()['id']}/download").content == downloads[batches[0]["id"]]


def test_scoring_rejects_changed_schema_or_model_without_saving_batches(trained_workspace):
    directory, frame, _, _, run = trained_workspace
    scoring = frame.iloc[:20].drop(columns="outcome").copy()
    invalid_numeric = scoring.copy()
    invalid_numeric.loc[0, "age"] = "not-a-number"
    duplicate_ids = scoring.copy()
    duplicate_ids.loc[1, "case_id"] = duplicate_ids.loc[0, "case_id"]
    missing_id = scoring.copy()
    missing_id.loc[0, "case_id"] = ""
    cases = [(scoring.drop(columns="annual_income"), "logistic", "raw"),
             (scoring.drop(columns="case_id"), "logistic", "raw"),
             (invalid_numeric, "logistic", "raw"), (duplicate_ids, "logistic", "raw"),
             (missing_id, "logistic", "raw"), (scoring, "random_forest", "raw"),
             (scoring, "logistic", "unknown")]
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        for candidate, model, calibration in cases:
            response = post_csv(client, f"/api/runs/{run['id']}/predict", csv_bytes(candidate),
                                model=model, calibration=calibration)
            assert response.status_code == 422, response.text
            assert isinstance(response.json()["detail"], str)
        assert client.get("/api/predictions").json() == {"items": []}


def test_tampered_model_is_rejected_before_deserialization(trained_workspace, monkeypatch):
    directory, frame, _, _, run = trained_workspace
    (directory / "runs" / f"{run['id']}.joblib").write_bytes(b"forged serialized model")
    calls = []

    def forbidden_load(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Untrusted model bytes reached deserialization")

    monkeypatch.setattr("service.api.joblib.load", forbidden_load)
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        response = post_csv(client, f"/api/runs/{run['id']}/predict", csv_bytes(frame.iloc[:10]),
                            model="logistic", calibration="raw")
        assert response.status_code == 422, response.text
        assert "校验" in response.json()["detail"]
        assert calls == []
        assert client.get("/api/predictions").json() == {"items": []}


def test_modified_dataset_cannot_be_used_for_a_new_analysis(trained_workspace):
    directory, _, _, dataset, run = trained_workspace
    (directory / "datasets" / f"{dataset['id']}.csv").write_bytes(b"changed,data\n1,2\n")
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        response = client.post("/api/runs", json=run["config"])
        assert response.status_code == 422, response.text
        assert "校验" in response.json()["detail"]
        assert len(client.get("/api/runs").json()["items"]) == 1


def test_blank_analysis_names_are_rejected_without_queuing_runs(trained_workspace):
    directory, _, _, _, run = trained_workspace
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        for name in ("", " ", "\t\n ", "\u3000"):
            response = client.post("/api/runs", json=dict(run["config"], name=name))
            assert response.status_code == 422, response.text
            assert isinstance(response.json()["detail"], str)
        assert client.get("/api/runs").json()["items"] == [run]


def test_analysis_name_is_trimmed_in_saved_configuration_and_after_restart(trained_workspace):
    directory, _, _, _, run = trained_workspace
    expected = "Trimmed analysis name"
    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as client:
        response = client.post("/api/runs", json=dict(run["config"], name=f" \t{expected}\n "))
        assert response.status_code == 202, response.text
        created = response.json()
        assert created["name"] == created["config"]["name"] == expected
        finished = completed_run(client, created["id"])
        assert finished["name"] == finished["config"]["name"] == expected

    with TestClient(create_app(directory, demo_mode=True, seed_demo=False)) as restarted:
        persisted = restarted.get(f"/api/runs/{created['id']}").json()
        assert persisted["name"] == persisted["config"]["name"] == expected
        assert persisted["status"] == "completed"


def test_restart_marks_interrupted_jobs_failed_without_retraining(tmp_path):
    app = create_app(tmp_path, demo_mode=True, seed_demo=False)
    with TestClient(app):
        for number, status in enumerate(("queued", "running", "failed"), 1):
            app.state.store.put("runs", {
                "id": f"{number:032x}", "name": status, "dataset_id": UNKNOWN_ID,
                "dataset_name": "interrupted.csv", "status": status, "progress": 10,
                "message": "Original message", "error": "Original failure" if status == "failed" else None,
                "created_at": "2026-01-01T00:00:00+00:00", "completed_at": None,
                "config": {}, "result": None,
            })
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        for number in (1, 2):
            identifier = f"{number:032x}"
            run = client.get(f"/api/runs/{identifier}").json()
            assert run["status"] == "failed" and run["error"] and run["completed_at"]
            assert run["result"] is None
            assert client.get(f"/api/runs/{identifier}/report").status_code == 409
            assert post_csv(client, f"/api/runs/{identifier}/predict", b"a\n1\n",
                            model="logistic", calibration="raw").status_code == 409
        assert client.get(f"/api/runs/{3:032x}").json()["error"] == "Original failure"


def test_authentication_setup_cookies_logout_and_persistence(tmp_path):
    with TestClient(create_app(tmp_path, demo_mode=False, seed_demo=False)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/auth/status").json() == {
            "configured": False, "authenticated": False, "demo_mode": False
        }
        for path in ("/api/dashboard", "/api/datasets", "/api/runs", "/api/predictions",
                     f"/api/datasets/{UNKNOWN_ID}/download", f"/api/runs/{UNKNOWN_ID}/report",
                     f"/api/runs/{UNKNOWN_ID}/template", f"/api/predictions/{UNKNOWN_ID}/download"):
            assert client.get(path).status_code == 401, path
        # Unauthorized malformed bodies must be rejected before JSON/multipart parsing.
        assert client.post("/api/runs", content=b"invalid JSON",
                           headers={"Content-Type": "application/json"}).status_code == 401
        assert client.post("/api/datasets", content=b"malformed multipart",
                           headers={"Content-Type": "multipart/form-data"}).status_code == 401
        assert client.post("/api/datasets/demo").status_code == 401
        assert client.post(f"/api/runs/{UNKNOWN_ID}/predict").status_code == 401
        # Public, idempotent logout also removes stale or expired browser cookies.
        stale_logout = client.post("/api/auth/logout", headers={"Cookie": "aware_session=expired-token"})
        assert stale_logout.status_code == 200
        assert "max-age=0" in stale_logout.headers["set-cookie"].lower()
        assert client.post("/api/auth/setup", json={"password": "short"}).status_code == 422
        response = client.post("/api/auth/setup", json={"password": PASSWORD})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
        token = client.cookies.get("aware_session")
        assert token and PASSWORD not in response.text
        assert client.get("/api/auth/status").json()["authenticated"] is True
        assert client.get("/api/datasets").status_code == 200
        assert client.post("/api/auth/setup", json={"password": PASSWORD + "changed"}).status_code == 409
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/status").json()["authenticated"] is False
        assert client.get("/api/datasets", headers={"Cookie": f"aware_session={token}"}).status_code == 401
        assert client.post("/api/auth/login", json={"password": PASSWORD + "wrong"}).status_code == 401
        assert client.post("/api/auth/login", json={"password": PASSWORD}).status_code == 200
        assert client.cookies.get("aware_session") != token
        assert client.get("/api/dashboard").headers["cache-control"] == "no-store"

    with TestClient(create_app(tmp_path, demo_mode=False, seed_demo=False)) as restarted:
        assert restarted.get("/api/auth/status").json() == {
            "configured": True, "authenticated": False, "demo_mode": False
        }
        assert restarted.post("/api/auth/setup", json={"password": PASSWORD}).status_code == 409
        assert restarted.post("/api/auth/login", json={"password": PASSWORD}).status_code == 200


def test_hostile_origins_cannot_set_up_or_mutate_an_authenticated_workspace(tmp_path):
    hostile = {"Origin": "https://attacker.example"}
    with TestClient(create_app(tmp_path, demo_mode=False, seed_demo=False)) as client:
        assert client.post("/api/auth/setup", json={"password": PASSWORD}, headers=hostile).status_code == 403
        assert client.get("/api/auth/status").json()["configured"] is False
        assert client.post("/api/auth/setup", json={"password": PASSWORD},
                           headers={"Origin": "http://testserver"}).status_code == 200
        assert client.post("/api/datasets/demo", headers=hostile).status_code == 403
        assert client.post("/api/auth/logout", headers=hostile).status_code == 403
        assert client.get("/api/auth/status").json()["authenticated"] is True
        assert client.get("/api/datasets").json() == {"items": []}
        assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 400


def test_demo_access_and_first_setup_are_local_only(tmp_path):
    with TestClient(create_app(tmp_path / "demo", demo_mode=True, seed_demo=False),
                    client=("198.51.100.20", 45000)) as remote:
        assert remote.get("/api/auth/status").status_code == 403
        assert remote.get("/api/datasets").status_code == 403
    with TestClient(create_app(tmp_path / "private", demo_mode=False, seed_demo=False),
                    client=("198.51.100.20", 45000)) as remote:
        assert remote.post("/api/auth/setup", json={"password": PASSWORD}).status_code == 403
        assert remote.get("/api/auth/status").json()["configured"] is False


def test_explicit_demo_import_is_persisted_and_labeled_synthetic(tmp_path):
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        assert client.get("/api/auth/status").json() == {
            "configured": True, "authenticated": True, "demo_mode": True
        }
        assert client.post("/api/auth/setup", json={"password": PASSWORD}).status_code == 409
        response = client.post("/api/datasets/demo")
        assert response.status_code == 201, response.text
        dataset = response.json()
        assert dataset["source"] == "demo" and "模拟" in dataset["name"]
        assert dataset["rows"] >= 500
        assert client.get(f"/api/datasets/{dataset['id']}").json() == dataset
        downloaded = pd.read_csv(io.BytesIO(client.get(f"/api/datasets/{dataset['id']}/download").content))
        assert len(downloaded) == dataset["rows"]
        assert set(downloaded["outcome"]) == {0, 1}


def test_secure_cookie_option_and_failed_login_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("AWARE_SECURE_COOKIE", "1")
    with TestClient(create_app(tmp_path, demo_mode=False, seed_demo=False), base_url="https://testserver") as client:
        response = client.post("/api/auth/setup", json={"password": PASSWORD})
        assert response.status_code == 200
        assert "secure" in response.headers["set-cookie"].lower()
        client.post("/api/auth/logout")
        for _ in range(8):
            assert client.post("/api/auth/login", json={"password": PASSWORD + "wrong"}).status_code == 401
        assert client.post("/api/auth/login", json={"password": PASSWORD}).status_code == 429


@pytest.mark.parametrize("content", [
    b"", b"only_header\n", b"x\n\xff\n", b"x,x\n1,2\n", b"x, x \n1,2\n",
    b"x,\n1,2\n", b"x,y\n1,2,3\n", b'x\n"unterminated\n', b"x\n\x00\n",
    (",".join(f"x{i}" for i in range(101)) + "\n" + ",".join("1" for _ in range(101))).encode(),
    b"x\n" + b"1\n" * 50_001,
])
def test_bad_csv_returns_readable_error_without_persisting_a_dataset(tmp_path, content):
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        response = post_csv(client, "/api/datasets", content)
        assert response.status_code == 422, response.text
        assert isinstance(response.json()["detail"], str) and response.json()["detail"]
        assert client.get("/api/datasets").json() == {"items": []}


def test_csv_bom_category_values_and_safe_uploaded_filename(tmp_path):
    content = "\ufeffcategory,value\nNA,?\nNone,\n".encode("utf-8")
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        response = client.post("/api/datasets", files={"file": ("../../credit.csv", content, "text/csv")})
        assert response.status_code == 201, response.text
        dataset = response.json()
        assert dataset["name"] == "credit.csv"
        assert dataset["columns"][0]["sample_values"] == ["NA", "None"]
        assert dataset["columns"][0]["missing_count"] == 0
        assert dataset["columns"][1]["missing_count"] == 2
        assert client.get(f"/api/datasets/{dataset['id']}/download").content == content


def test_file_size_boundary_and_content_length_limit(tmp_path):
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        # A large whitespace field keeps the exact-limit valid CSV response small.
        exact_limit = b"value\n1\n" + b" " * (MAX_UPLOAD - 8)
        response = post_csv(client, "/api/datasets", exact_limit)
        assert response.status_code == 201, response.text
        assert response.json()["rows"] == 2
        response = post_csv(client, "/api/datasets", exact_limit + b" ")
        assert response.status_code == 413, response.text
        response = client.post("/api/datasets", content=b"x",
                               headers={"Content-Length": str(MAX_UPLOAD + 65537)})
        assert response.status_code == 413, response.text
        assert len(client.get("/api/datasets").json()["items"]) == 1


@pytest.mark.parametrize("declared_length", [None, "4"])
def test_streaming_request_cannot_bypass_body_limit(tmp_path, declared_length):
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    with TestClient(create_app(tmp_path, demo_mode=True, seed_demo=False)) as client:
        response = client.post("/api/auth/login", headers=headers,
                               content=iter([b"{", b"x" * (MAX_UPLOAD + 65536), b"}"]))
        assert response.status_code == 413, response.text[:300]


def test_unknown_identifiers_and_traversal_never_read_arbitrary_files(tmp_path):
    secret = tmp_path / "outside.csv"
    secret.write_text("private-arbitrary-file-content", encoding="utf-8")
    with TestClient(create_app(tmp_path / "workspace", demo_mode=True, seed_demo=False)) as client:
        for identifier in (UNKNOWN_ID, "not-an-id", "%2e%2e%2foutside.csv", "%252e%252e%252foutside.csv"):
            for path in (f"/api/datasets/{identifier}", f"/api/datasets/{identifier}/download",
                         f"/api/runs/{identifier}", f"/api/runs/{identifier}/report",
                         f"/api/runs/{identifier}/template", f"/api/predictions/{identifier}/download"):
                response = client.get(path)
                assert response.status_code == 404, (path, response.text)
                assert secret.read_text() not in response.text
        response = client.get("/%2e%2e%2f%2e%2e%2fservice%2fsecurity.py")
        assert response.status_code == 404
