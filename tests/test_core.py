from datetime import date

import pytest

import main
import ml_service


def item(day, tile="T21LWF", cloud=10):
    return {
        "id": f"S2_{day}_{tile}",
        "properties": {"datetime": f"{day}T10:00:00Z", "s2:mgrs_tile": tile, "eo:cloud_cover": cloud},
        "assets": {"red": {"href": "red.tif"}, "nir": {"href": "nir.tif"}, "scl": {"href": "scl.tif"}},
    }


def test_pair_selector_uses_low_cloud_same_tile_pair():
    before = [item("2024-06-10", cloud=20), item("2024-06-12", cloud=5)]
    after = [item("2025-06-10", cloud=8), item("2025-06-15", cloud=40)]
    chosen_before, chosen_after = main._select_pair_from_windows(before, after, minimum_gap=30)
    assert chosen_before["id"] == "S2_2024-06-12_T21LWF"
    assert chosen_after["id"] == "S2_2025-06-10_T21LWF"


def test_pair_selector_rejects_tile_mismatch():
    with pytest.raises(main.HTTPException) as error:
        main._select_pair_from_windows([item("2024-06-10", "T21LWF")], [item("2025-06-10", "T22LWF")], 30)
    assert error.value.status_code == 404


def test_pair_selector_rejects_too_small_gap():
    with pytest.raises(main.HTTPException) as error:
        main._select_pair_from_windows([item("2025-01-01")], [item("2025-01-20")], 30)
    assert error.value.status_code == 404


def test_bbox_must_be_valid_and_overlap_boundary():
    boundary = main.shape({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
    request = main.SentinelAnalysisRequest(
        bbox=[0.2, 0.2, 0.3, 0.3], start_date=date(2024, 1, 1), end_date=date(2025, 1, 1)
    )
    assert main._validate_bbox(request, boundary) == (0.2, 0.2, 0.3, 0.3)
    request.bbox = [2, 2, 2.1, 2.1]
    with pytest.raises(main.HTTPException) as error:
        main._validate_bbox(request, boundary)
    assert error.value.status_code == 400


def sample_rows():
    rows = []
    for i in range(12):
        stable = i % 2 == 0
        rows.append({
            "label": "stable_forest" if stable else "change_candidate",
            "ndvi_before": 0.8 if stable else 0.8,
            "ndvi_after": 0.78 if stable else 0.35,
            "ndvi_drop": 0.02 if stable else 0.45,
            "red_before": 0.08, "nir_before": 0.42,
            "red_after": 0.09 if stable else 0.25,
            "nir_after": 0.4 if stable else 0.27,
            "valid_coverage_percent": 95.0,
            "candidate_area_hectares": 0.0 if stable else 2.0,
        })
    return rows


def test_ml_rejects_too_few_rows():
    with pytest.raises(ValueError, match="At least 10"):
        ml_service._rows_to_xy(sample_rows()[:9])


def test_ml_training_reports_metrics_and_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(ml_service, "MODEL_PATH", model_path)
    metrics = ml_service.train(sample_rows())
    assert metrics["train_rows"] + metrics["test_rows"] == 12
    assert 0 <= metrics["macro_f1"] <= 1
    prediction = ml_service.predict({key: value for key, value in sample_rows()[0].items() if key != "label"})
    assert prediction["label"] in {"stable_forest", "change_candidate"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=0.001)
    assert prediction["validation"]["warning"].startswith("Validation is only")


def test_ml_rejects_single_class():
    rows = sample_rows()
    for row in rows:
        row["label"] = "stable_forest"
    with pytest.raises(ValueError, match="at least two"):
        ml_service._rows_to_xy(rows)

def test_grouped_validation_uses_group_holdout(monkeypatch, tmp_path):
    model_path = tmp_path / "grouped.joblib"
    monkeypatch.setattr(ml_service, "MODEL_PATH", model_path)
    rows = []
    for group_index in range(4):
        for j in range(4):
            stable = (group_index + j) % 2 == 0
            rows.append({
                "group": f"site_{group_index}",
                "label": "stable_forest" if stable else "change_candidate",
                "ndvi_before": .8, "ndvi_after": .78 if stable else .35,
                "ndvi_drop": .02 if stable else .45,
                "red_before": .08, "nir_before": .42,
                "red_after": .09 if stable else .25, "nir_after": .4 if stable else .27,
                "valid_coverage_percent": 95.0,
                "candidate_area_hectares": 0.0 if stable else 2.0,
            })
    metrics = ml_service.train(rows)
    assert metrics["validation_method"] == "grouped_holdout"

def test_priority_increases_with_persistence_and_quality():
    low = main._priority_from_evidence(0.10, 1, 5, 55, "medium")
    high = main._priority_from_evidence(0.35, 4, 5, 95, "low")
    assert high[1] > low[1]
    assert high[0] in {"medium", "high"}


def test_seasonal_month_distance_wraps_december_to_january():
    assert main._month_distance(12, 1) == 1
    assert main._month_distance(1, 12) == 1
    assert main._month_distance(6, 9) == 3


def test_latest_valid_observation_ignores_invalid_trailing_observation():
    observations = [
        {"date": "2025-01-01", "valid": True, "ndvi_drop_from_baseline": 0.02},
        {"date": "2025-02-01", "valid": True, "ndvi_drop_from_baseline": 0.18},
        {"date": "2025-03-01", "valid": False, "ndvi_drop_from_baseline": 0.90},
    ]
    latest = main._latest_valid_observation(observations)
    assert latest["date"] == "2025-02-01"
    assert latest["ndvi_drop_from_baseline"] == 0.18


def test_review_import_populates_metadata_and_edit_replaces_duplicate(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    history_file = tmp_path / "history.json"
    labels_file = tmp_path / "labels.json"
    history_file.write_text(__import__("json").dumps([{
        "job_id": "job-1", "mode": "sentinel2", "before_date": "2024-06-01", "after_date": "2025-06-01",
        "candidate_centroid": {"latitude": -7.1, "longitude": -56.2}, "candidate_area_hectares": 2.5,
        "valid_pixel_coverage_percent": 94, "mean_ndvi_before": .8, "mean_ndvi_after": .4, "mean_ndvi_drop": .4,
        "mean_red_before": .08, "mean_nir_before": .42, "mean_red_after": .25, "mean_nir_after": .27,
        "severity": "medium", "activity_label": "Possible vegetation decrease", "map_url": "/evidence/job-1.png"
    }]))
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    monkeypatch.setattr(main, "LABELS_FILE", labels_file)
    client = TestClient(main.app)
    imported = client.post("/labels/from-analysis/job-1", params={"group": "site-01"})
    assert imported.status_code == 200
    assert imported.json()["before_date"] == "2024-06-01"
    assert imported.json()["latitude"] == -7.1
    update = imported.json() | {"label": "vegetation_loss", "review_status": "accepted", "reviewer_confidence": "high"}
    assert client.post("/labels", json=update).status_code == 200
    assert client.post("/labels", json=update).status_code == 200
    assert client.get("/labels").json()["count"] == 1
    assert client.get("/labels/export").json()["accepted_count"] == 1


def test_ml_readiness_is_explicit_when_zero_labels(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main, "LABELS_FILE", tmp_path / "labels.json")
    client = TestClient(main.app)
    readiness = client.get("/ml/readiness")
    assert readiness.status_code == 200
    assert readiness.json()["ready"] is False
    assert readiness.json()["accepted_count"] == 0
    training = client.post("/train-ml-from-labels")
    assert training.status_code == 400
    assert "optional" in training.json()["detail"]


def test_evidence_report_returns_saved_analysis(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    history_file = tmp_path / "history.json"
    history_file.write_text(__import__("json").dumps([{"job_id": "saved-1", "mode": "sentinel2_timeseries", "status": "no_persistent_change"}]))
    monkeypatch.setattr(main, "HISTORY_FILE", history_file)
    report = TestClient(main.app).get("/evidence-report/saved-1")
    assert report.status_code == 200
    assert report.json()["result"]["job_id"] == "saved-1"
