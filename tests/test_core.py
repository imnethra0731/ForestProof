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
      































































