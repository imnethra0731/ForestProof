"""ForestProof Streamlit dashboard.

The main path fetches real Sentinel-2 scenes through the FastAPI backend.
Manual uploads remain available as an optional RGB pixel-comparison test.
"""

from __future__ import annotations

from datetime import date, timedelta
import os
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="ForestProof | Forest Change Review",
    page_icon="🌿",
    layout="wide",
)

def configured_api_url() -> str:
    value = os.environ.get("FORESTPROOF_API_URL")
    try:
        value = st.secrets.get("FORESTPROOF_API_URL", value)
    except Exception:
        # Local runs without a Streamlit secrets file should still work.
        pass
    return (value or "http://127.0.0.1:8000").strip().rstrip("/")


API = configured_api_url()
REQUEST_TIMEOUT = 240

st.title("ForestProof")
st.caption(
    "Satellite imagery change-review prototype · Sentinel-2 L2A · Jamanxim National Forest, Pará, Brazil"
)
st.warning(
    "ForestProof identifies candidate vegetation change for human review. "
    "It does not prove illegal activity or classify construction, roads, or mining."
)


def api_get(path: str, timeout: int = 30) -> Any:
    response = requests.get(f"{API}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def show_api_error(exc: Exception) -> None:
    detail = str(exc)
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            detail = response.json().get("detail", detail)
    except Exception:
        pass
    st.error(detail)


def build_area_map(
    area_data: dict[str, Any],
    bbox: list[float],
    marker: dict[str, float] | None = None,
    reference_feature: dict[str, Any] | None = None,
) -> folium.Map:
    west, south, east, north = bbox
    fmap = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.GeoJson(
        area_data["boundary_geojson"],
        name="Official ICMBio protected-area boundary",
        style_function=lambda _: {"color": "#12805c", "weight": 2, "fillColor": "#32a852", "fillOpacity": 0.08},
        tooltip=area_data.get("name", "Protected forest boundary"),
    ).add_to(fmap)
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#d62728",
        weight=2,
        fill=True,
        fill_color="#d62728",
        fill_opacity=0.12,
        tooltip="Selected analysis box",
    ).add_to(fmap)
    if marker:
        folium.Marker(
            [marker["latitude"], marker["longitude"]],
            tooltip="Selected site / change centroid",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(fmap)
    if reference_feature:
        folium.GeoJson(
            reference_feature,
            name="INPE PRODES historical reference polygon",
            style_function=lambda _: {"color": "#ff8c00", "weight": 3, "fillColor": "#ff8c00", "fillOpacity": 0.3},
            tooltip="Official INPE PRODES historical reference feature",
        ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


def render_result(result: dict[str, Any], area_data: dict[str, Any] | None = None) -> None:
    st.subheader("Analysis result")
    if result.get("mode") == "sentinel2":
        c1, c2, c3 = st.columns(3)
        area_value = result.get("candidate_area_hectares")
        decrease_value = result.get("candidate_vegetation_decrease_percent_of_baseline")
        c1.metric("Candidate area", f"{area_value:,.3f} ha" if area_value is not None else "N/A")
        c2.metric(
            "Decrease vs baseline forest-like pixels",
            f"{decrease_value:.2f}%" if decrease_value is not None else "N/A",
        )
        c3.metric("Usable paired pixels", f"{result.get('valid_pixel_coverage_percent', 0):.1f}%")
        st.write(f"**Status:** `{result.get('status', 'unknown')}`")
        st.write(f"**Possible change:** {result.get('activity_label', 'Not classified')}")
        st.write(
            f"**Dates:** {result.get('before_date')} → {result.get('after_date')}  "
            f"| **Source:** {result.get('sensor')}"
        )
        st.write(
            f"**Scenes:** `{result.get('before_scene_id')}` → `{result.get('after_scene_id')}`"
        )
        st.write(
            f"**Scene cloud metadata:** before {result.get('before_scene_cloud_cover_percent')}%, "
            f"after {result.get('after_scene_cloud_cover_percent')}%. This is scene-wide metadata."
        )
        if result.get("candidate_centroid"):
            location = result["candidate_centroid"]
            st.write(
                f"**Candidate centroid:** {location['latitude']:.6f}, {location['longitude']:.6f}"
            )
        elif result.get("status") == "insufficient_image_quality":
            st.write("**Candidate centroid:** Not assessed because image quality was insufficient.")
        else:
            st.write("**Candidate centroid:** No candidate pixels above the selected rule.")
        st.info(result.get("explanation", "No explanation provided."))
        st.caption(
            "Method: baseline NDVI ≥ 0.50, after NDVI < 0.50, and NDVI decrease ≥ 0.20. "
            "Cloud/shadow screening uses Sentinel-2 SCL classes. These prototype thresholds need local validation."
        )
        reference = result.get("prodes_reference")
        if reference:
            st.markdown("**Independent historical reference (INPE PRODES)**")
            st.write(
                f"Year {reference.get('year')} · {reference.get('class_label')} · "
                f"reference mapped area {reference.get('area_hectares')} ha · "
                f"satellite image date {reference.get('image_date')}"
            )
            overlap = reference.get("ndvi_candidate_overlap_percent")
            if overlap is None:
                st.write("Satellite-candidate overlap with the reference: not assessed because image quality was insufficient.")
            else:
                st.write(f"Reference polygon covered by the NDVI candidate: {overlap:.2f}%")
            st.caption(reference.get("legal_status_note", ""))
            st.write(f"Reference source: {reference.get('source_url')}")
        st.caption(result.get("persistence_check", ""))
        with st.expander("Evidence, sources, and limitations"):
            st.write(f"Protected boundary: {result.get('boundary_name')}")
            st.write(f"Boundary source: {result.get('boundary_source_url')}")
            st.write(f"Imagery catalog: {result.get('imagery_catalog_url')}")
            st.write(f"Before scene: {result.get('before_source_url') or 'Catalog item URL not supplied'}")
            st.write(f"After scene: {result.get('after_source_url') or 'Catalog item URL not supplied'}")
            for limitation in result.get("limitations", []):
                st.write(f"- {limitation}")
        if area_data and result.get("bbox_wsen"):
            st.subheader("Location on the official boundary map")
            marker = result.get("candidate_centroid")
            reference_feature = result.get("prodes_reference", {}).get("feature") if result.get("prodes_reference") else None
            st_folium(
                build_area_map(area_data, result["bbox_wsen"], marker, reference_feature),
                height=430,
                use_container_width=True,
                key=f"area-map-{result.get('job_id')}",
            )
    else:
        st.metric("Different-looking pixels", f"{result.get('changed_pixel_percent', 0):.2f}%")
        st.write(f"**Status:** `{result.get('status')}`")
        st.info(result.get("explanation", ""))
        st.caption(
            "Manual uploads are an optional RGB pixel-difference test. They do not contain verified location, "
            "satellite bands, NDVI, or a protected-area boundary."
        )

    if result.get("map_url"):
        try:
            image_response = requests.get(f"{API}{result['map_url']}", timeout=30)
            image_response.raise_for_status()
            st.subheader("Image evidence")
            st.image(image_response.content, use_container_width=True)
        except requests.RequestException as exc:
            st.warning(f"Could not load the saved evidence preview: {exc}")


try:
    health = api_get("/health")
    area_data = api_get("/area/jamanxim")
except Exception as exc:
    st.error(
        "The backend is not reachable. Open a second Command Prompt in your ForestProof folder and run "
        "`python -m uvicorn main:app --reload`, then refresh this page."
    )
    st.code(str(exc))
    st.stop()

st.caption(
    f"Boundary: {area_data.get('name')} · Official mapped area: "
    f"{float(area_data.get('official_area_ha') or 0):,.0f} ha · "
    f"Boundary data: {area_data.get('boundary_source')}"
)

analysis_tab, timeseries_tab, label_tab, manual_tab, verify_tab, history_tab, monitor_tab = st.tabs(
    ["Satellite analysis", "Multi-date trend", "Label Review", "Manual upload test", "Verification + ML", "History", "Scheduled monitoring"]
)

with analysis_tab:
    st.subheader("Fetch and analyze real satellite imagery")
    st.write(
        "Click inside the green reserve to move the red analysis box, or enter its coordinates below. "
        "For a reproducible judge demo, load the verified historical PRODES case."
    )
    suggested = area_data["suggested_bbox"]
    today = date.today()
    before_center = today - timedelta(days=365 * 3)
    defaults = {
        "bbox_west": float(suggested[0]),
        "bbox_south": float(suggested[1]),
        "bbox_east": float(suggested[2]),
        "bbox_north": float(suggested[3]),
        "before_start": before_center - timedelta(days=45),
        "before_end": before_center + timedelta(days=45),
        "after_start": today - timedelta(days=90),
        "after_end": today,
        "cloud_limit": 80,
        "prodes_demo_case": None,
        "last_processed_map_click": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.button("Load verified Jamanxim historical demo case", type="secondary", use_container_width=True):
        try:
            with st.spinner("Loading the historical INPE PRODES reference feature…"):
                demo_case = api_get("/demo-case/jamanxim-prodes-2025", timeout=90)
            st.session_state["prodes_demo_case"] = demo_case
            west0, south0, east0, north0 = demo_case["suggested_bbox"]
            st.session_state["bbox_west"] = float(west0)
            st.session_state["bbox_south"] = float(south0)
            st.session_state["bbox_east"] = float(east0)
            st.session_state["bbox_north"] = float(north0)
            st.session_state["before_start"] = date(2024, 6, 1)
            st.session_state["before_end"] = date(2024, 9, 1)
            st.session_state["after_start"] = date(2025, 6, 1)
            st.session_state["after_end"] = date(2025, 9, 1)
            st.rerun()
        except Exception as exc:
            show_api_error(exc)

    demo_case = st.session_state.get("prodes_demo_case")
    if demo_case:
        st.info(
            f"Demo preset loaded: INPE PRODES mapped a 2025 feature labelled **{demo_case['sub_class']}** "
            f"({demo_case['area_hectares']} ha) inside the Jamanxim boundary. This reference is contextual evidence, "
            "not proof that an activity was illegal."
        )
    st.caption("The map uses public OpenStreetMap tiles and does not need a Google Maps API key. Internet access is required for map tiles.")
    st.info("Click a point inside the green boundary. The red box recentres around that point; the coordinates remain editable below.")
    west, south, east, north = [
        float(st.session_state[key])
        for key in ("bbox_west", "bbox_south", "bbox_east", "bbox_north")
    ]
    bbox = [west, south, east, north]
    center_marker = demo_case.get("reference_point") if demo_case else {
        "latitude": (south + north) / 2,
        "longitude": (west + east) / 2,
    }
    reference_feature = demo_case.get("feature") if demo_case else None
    try:
        map_event = st_folium(
            build_area_map(area_data, bbox, center_marker, reference_feature),
            height=430,
            use_container_width=True,
            key="study-area-map",
            returned_objects=["last_clicked"],
        )
        clicked = map_event.get("last_clicked") if map_event else None
        if clicked:
            click_token = f"{clicked['lat']:.5f},{clicked['lng']:.5f}"
            if click_token != st.session_state.get("last_processed_map_click"):
                half_width = 0.04
                st.session_state["bbox_west"] = float(clicked["lng"] - half_width)
                st.session_state["bbox_south"] = float(clicked["lat"] - half_width)
                st.session_state["bbox_east"] = float(clicked["lng"] + half_width)
                st.session_state["bbox_north"] = float(clicked["lat"] + half_width)
                st.session_state["prodes_demo_case"] = None
                st.session_state["last_processed_map_click"] = click_token
                st.rerun()
    except Exception as exc:
        st.warning(f"Could not display the interactive map: {exc}")

    with st.expander("Study area coordinates", expanded=True):
        left, right = st.columns(2)
        with left:
            west = st.number_input("West longitude", format="%.6f", step=0.005, key="bbox_west")
            south = st.number_input("South latitude", format="%.6f", step=0.005, key="bbox_south")
        with right:
            east = st.number_input("East longitude", format="%.6f", step=0.005, key="bbox_east")
            north = st.number_input("North latitude", format="%.6f", step=0.005, key="bbox_north")
    bbox = [float(west), float(south), float(east), float(north)]

    st.markdown("**Choose two periods to compare**")
    before_col, after_col = st.columns(2)
    with before_col:
        st.caption("BEFORE / BASELINE PERIOD")
        before_start_date = st.date_input("Before period starts", max_value=today, key="before_start")
        before_end_date = st.date_input("Before period ends", max_value=today, key="before_end")
    with after_col:
        st.caption("AFTER / COMPARISON PERIOD")
        after_start_date = st.date_input("After period starts", max_value=today, key="after_start")
        after_end_date = st.date_input("After period ends", max_value=today, key="after_end")
    cloud_limit = st.slider("Maximum scene-wide cloud metadata (%)", min_value=0, max_value=100, key="cloud_limit")
    d4, d5 = st.columns(2)
    min_gap = d4.slider("Minimum days between images", min_value=5, max_value=180, value=30)
    min_area = d5.number_input("Review-area threshold (hectares)", min_value=0.0, max_value=1000.0, value=0.5, step=0.1)

    request_payload = {
        "bbox": bbox,
        "start_date": min(before_start_date, after_start_date).isoformat(),
        "end_date": max(before_end_date, after_end_date).isoformat(),
        "before_start_date": before_start_date.isoformat(),
        "before_end_date": before_end_date.isoformat(),
        "after_start_date": after_start_date.isoformat(),
        "after_end_date": after_end_date.isoformat(),
        "prodes_reference_uuid": demo_case.get("uuid") if demo_case else None,
        "cloud_limit": float(cloud_limit),
        "min_gap_days": int(min_gap),
        "minimum_alert_area_ha": float(min_area),
        "interval_hours": 24,
    }
    if st.button("Fetch and analyze Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching the satellite catalog, reading image bands, and calculating the comparison…"):
                response = requests.post(
                    f"{API}/analyze-sentinel",
                    json=request_payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state["active_result"] = result
        except Exception as exc:
            show_api_error(exc)
    if st.session_state.get("active_result", {}).get("mode") == "sentinel2":
        render_result(st.session_state["active_result"], area_data)

with timeseries_tab:
    st.subheader("Multi-date vegetation trend")
    st.write(
        "This scans several same-tile Sentinel-2 observations, prefers a comparable seasonal window, "
        "and escalates only when a vegetation decrease persists. It is a trend summary, not a pixel-level area polygon."
    )
    ts_start = st.date_input("Trend start", value=date(2024, 1, 1), key="ts-start")
    ts_end = st.date_input("Trend end", value=today, key="ts-end")
    ts_min_obs = st.slider("Minimum observations", 3, 12, 4, key="ts-min-obs")
    ts_seasonal = st.checkbox("Prefer comparable seasonal months", True, key="ts-seasonal")
    if st.button("Analyze multi-date trend", type="primary", key="ts-run", use_container_width=True):
        try:
            with st.spinner("Searching multiple acquisitions and checking persistence…"):
                response = requests.post(
                    f"{API}/analyze-timeseries",
                    json={
                        "bbox": [float(st.session_state["bbox_west"]), float(st.session_state["bbox_south"]), float(st.session_state["bbox_east"]), float(st.session_state["bbox_north"])],
                        "start_date": ts_start.isoformat(), "end_date": ts_end.isoformat(),
                        "cloud_limit": float(st.session_state.get("cloud_limit", 80)),
                        "minimum_observations": int(ts_min_obs), "seasonal_months_only": bool(ts_seasonal),
                    }, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                st.session_state["timeseries_result"] = response.json()
        except Exception as exc:
            show_api_error(exc)
    if st.session_state.get("timeseries_result"):
        ts_result = st.session_state["timeseries_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", ts_result.get("status"))
        c2.metric("Persistence count", ts_result.get("persistence_observation_count"))
        c3.metric("Priority", f"{ts_result.get('inspection_priority')} ({ts_result.get('priority_score')}/100)")
        st.write(f"**Severity:** {ts_result.get('severity')} · **Seasonal risk:** {ts_result.get('seasonal_risk')}")
        st.info(ts_result.get("explanation", ""))
        st.dataframe(ts_result.get("observations", []), use_container_width=True)
        st.caption(f"Evidence report: {API}/evidence-report/{ts_result.get('job_id')}")

with label_tab:
    st.subheader("Label Review — research/admin tool")
    st.warning("Review evidence before accepting a label. Accepted labels are training data; uncertain and rejected records remain in the audit list but are excluded from ML training.")
    try:
        label_data = api_get("/labels")
        all_items = label_data.get("items", [])
    except Exception as exc:
        all_items = []
        show_api_error(exc)

    # Summary and category counts make the review state visible even with zero samples.
    counts = {status: sum(1 for item in all_items if item.get("review_status") == status) for status in ("accepted", "rejected", "uncertain")}
    pending = counts["uncertain"]
    summary_cols = st.columns(5)
    summary_cols[0].metric("Total samples", len(all_items))
    summary_cols[1].metric("Pending reviews", pending)
    summary_cols[2].metric("Accepted", counts["accepted"])
    summary_cols[3].metric("Rejected", counts["rejected"])
    summary_cols[4].metric("Uncertain", counts["uncertain"])
    category_counts = {}
    for item in all_items:
        if item.get("review_status") == "accepted":
            category_counts[item.get("label", "unknown")] = category_counts.get(item.get("label", "unknown"), 0) + 1
    if category_counts:
        st.caption("Accepted samples by label: " + " · ".join(f"{label}: {count}" for label, count in sorted(category_counts.items())))
    else:
        st.info("No accepted labels yet. The application remains fully usable; ML training is optional.")

    filter_cols = st.columns(2)
    filter_label = filter_cols[0].selectbox("Filter label", ["all"] + sorted({str(x.get("label", "unknown")) for x in all_items}), key="review-filter-label")
    filter_status = filter_cols[1].selectbox("Filter status", ["all", "uncertain", "accepted", "rejected"], key="review-filter-status")
    filtered_items = [x for x in all_items if (filter_label == "all" or x.get("label") == filter_label) and (filter_status == "all" or x.get("review_status") == filter_status)]
    st.caption(f"Showing {len(filtered_items)} of {len(all_items)} samples.")

    active_satellite = st.session_state.get("active_result", {})
    if active_satellite.get("mode") == "sentinel2":
        import_group = st.text_input("Group for imported satellite sample", value="jamanxim_review_site", key="import-label-group")
        if st.button("Import latest real Sentinel-2 result for review", key="import-satellite-label", use_container_width=True):
            try:
                response = requests.post(f"{API}/labels/from-analysis/{active_satellite.get('job_id')}", params={"group": import_group}, timeout=30)
                response.raise_for_status()
                imported = response.json()
                st.session_state["review-sample-select"] = imported.get("sample_id")
                st.success(f"Imported {imported.get('sample_id')} as uncertain. Select it below to review and edit.")
                st.rerun()
            except Exception as exc:
                show_api_error(exc)

    sample_options = ["new_sample"] + [str(item.get("sample_id")) for item in filtered_items if item.get("sample_id")]
    selected_id = st.selectbox("Review sample", sample_options, key="review-sample-select")
    selected = next((item for item in all_items if item.get("sample_id") == selected_id), None)
    if selected:
        features_existing = selected.get("features", {})
        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("Before NDVI", f"{float(features_existing.get('ndvi_before', 0)):.4f}")
        ev2.metric("After NDVI", f"{float(features_existing.get('ndvi_after', 0)):.4f}")
        ev3.metric("NDVI difference", f"{float(features_existing.get('ndvi_drop', 0)):.4f}")
        st.write(f"**Dates:** {selected.get('before_date') or 'not recorded'} → {selected.get('after_date') or 'not recorded'}")
        st.write(f"**Coordinates:** {selected.get('latitude') if selected.get('latitude') is not None else 'not recorded'}, {selected.get('longitude') if selected.get('longitude') is not None else 'not recorded'}")
        st.write(f"**Severity:** {selected.get('severity', 'not assigned')} · **Possible change:** {selected.get('possible_change', 'candidate vegetation change')}")
        if selected.get("map_url"):
            try:
                evidence_response = requests.get(f"{API}{selected['map_url']}", timeout=30)
                evidence_response.raise_for_status()
                st.image(evidence_response.content, caption="Saved satellite evidence preview", use_container_width=True)
            except requests.RequestException:
                st.warning("The saved evidence preview is currently unavailable. Review the scene links and metadata instead.")
        else:
            st.info("No saved visual preview is attached to this sample. The metadata and numeric features are still available for review.")
        if selected.get("source_job_id"):
            st.caption(f"Imported from satellite analysis job: {selected['source_job_id']}")

    feature_names = ["ndvi_before", "ndvi_after", "ndvi_drop", "red_before", "nir_before", "red_after", "nir_after", "valid_coverage_percent", "candidate_area_hectares"]
    defaults = {"ndvi_before": 0.80, "ndvi_after": 0.75, "ndvi_drop": 0.05, "red_before": 0.08, "nir_before": 0.42, "red_after": 0.09, "nir_after": 0.40, "valid_coverage_percent": 95.0, "candidate_area_hectares": 0.0}
    existing_features = (selected or {}).get("features", {})
    if selected:
        sample_id_value = str(selected.get("sample_id", ""))
        group_value = str(selected.get("group", "unassigned_site"))
        source_value = str(selected.get("label_source", "user_review"))
        status_value = str(selected.get("review_status", "uncertain"))
        confidence_value = str(selected.get("reviewer_confidence", "medium"))
        label_value = str(selected.get("label", "unknown"))
        notes_value = str(selected.get("notes", ""))
        lat_value = selected.get("latitude")
        lon_value = selected.get("longitude")
        before_value = selected.get("before_date") or ""
        after_value = selected.get("after_date") or ""
    else:
        sample_id_value, group_value, source_value, status_value, confidence_value, label_value, notes_value = "", "jamanxim_review_site", "public satellite review / PRODES context", "uncertain", "medium", "unknown", ""
        lat_value = lon_value = None
        before_value = after_value = ""
    st.markdown("**Review decision** — imported metadata is populated automatically; normally you only change the label, status, confidence, and optional notes.")
    label_name = st.selectbox("Label", ["stable_forest", "vegetation_loss", "bare_soil", "water_or_flooding", "burned_area", "road", "unknown"], index=["stable_forest", "vegetation_loss", "bare_soil", "water_or_flooding", "burned_area", "road", "unknown"].index(label_value) if label_value in ["stable_forest", "vegetation_loss", "bare_soil", "water_or_flooding", "burned_area", "road", "unknown"] else 6, key=f"review-label-{selected_id}")
    label_status = st.selectbox("Review status", ["accepted", "uncertain", "rejected"], index=["accepted", "uncertain", "rejected"].index(status_value) if status_value in ["accepted", "uncertain", "rejected"] else 1, key=f"review-status-{selected_id}")
    confidence = st.selectbox("Reviewer confidence", ["high", "medium", "low"], index=["high", "medium", "low"].index(confidence_value) if confidence_value in ["high", "medium", "low"] else 1, key=f"review-confidence-{selected_id}")
    notes_value = st.text_area("Notes (optional)", value=notes_value, key=f"review-notes-{selected_id}")
    if selected:
        st.caption(f"Sample ID: {sample_id_value} · Group: {group_value} · Source: {source_value} · Dates: {before_value} → {after_value}")
    else:
        sample_id_value = st.text_input("Sample ID", placeholder="site_001_patch_01", key="new-review-sample-id")
        group_value = st.text_input("Group/site ID", value=group_value, key="new-review-group")
        source_value = st.text_input("Label source", value=source_value, key="new-review-source")
        before_value = st.text_input("Before date (optional)", key="new-review-before")
        after_value = st.text_input("After date (optional)", key="new-review-after")
    fcols = st.columns(3)
    features = {}
    for i, feature in enumerate(feature_names):
        features[feature] = fcols[i % 3].number_input(feature, value=float(existing_features.get(feature, defaults[feature])), key=f"review-feature-{selected_id}-{feature}")
    if st.button("Save label / update existing review", type="primary", key=f"save-label-{selected_id}", use_container_width=True):
        if not sample_id_value or not group_value:
            st.error("Sample ID and group/site ID are required.")
        else:
            try:
                payload = {"sample_id": sample_id_value, "group": group_value, "label": label_name, "review_status": label_status, "latitude": lat_value, "longitude": lon_value, "before_date": before_value or None, "after_date": after_value or None, "label_source": source_value, "reviewer_confidence": confidence, "notes": notes_value, "features": features}
                response = requests.post(f"{API}/labels", json=payload, timeout=30)
                response.raise_for_status()
                st.success("Review saved. Existing sample IDs are updated instead of duplicated.")
                st.rerun()
            except Exception as exc:
                show_api_error(exc)

    st.divider()
    try:
        readiness = api_get("/ml/readiness")
        if readiness.get("ready"):
            st.success(f"ML ready: {readiness.get('accepted_count')} accepted samples across {len(readiness.get('available_classes', []))} classes.")
        else:
            st.info(f"ML is optional and not ready yet: {readiness.get('accepted_count', 0)} accepted samples; available classes: {', '.join(readiness.get('available_classes', [])) or 'none'}. No accuracy is shown until genuine validation is possible.")
        if st.button("Train from accepted genuine labels", key="train-reviewed", disabled=not readiness.get("ready", False), use_container_width=True):
            response = requests.post(f"{API}/train-ml-from-labels", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            st.session_state["reviewed_train_result"] = response.json()
        if st.session_state.get("reviewed_train_result"):
            st.json(st.session_state["reviewed_train_result"].get("metrics", st.session_state["reviewed_train_result"]))
    except requests.HTTPError as exc:
        show_api_error(exc)
    except Exception as exc:
        show_api_error(exc)

with manual_tab:
    st.subheader("Optional manual image test")
    st.write(
        "Upload two same-sized PNG/JPG files to test the upload path. This is separate from satellite monitoring. "
        "The backend compares RGB pixels only; it cannot determine location or vegetation loss from these files."
    )
    before_file = st.file_uploader("Before image", type=["png", "jpg", "jpeg"], key="manual-before")
    after_file = st.file_uploader("After image", type=["png", "jpg", "jpeg"], key="manual-after")
    if st.button("Compare uploaded images", key="manual-run", use_container_width=True):
        if not before_file or not after_file:
            st.error("Choose both image files first.")
        else:
            try:
                with st.spinner("Comparing the RGB pixels…"):
                    response = requests.post(
                        f"{API}/upload-manual",
                        files={
                            "before_file": (before_file.name, before_file.getvalue(), before_file.type or "image/png"),
                            "after_file": (after_file.name, after_file.getvalue(), after_file.type or "image/png"),
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state["active_manual_result"] = result
            except Exception as exc:
                show_api_error(exc)
    if st.session_state.get("active_manual_result"):
        render_result(st.session_state["active_manual_result"])

with verify_tab:
    st.subheader("Georeferenced verification and ML")
    st.write(
        "Upload a real georeferenced UAV orthomosaic when one is available. "
        "The backend rejects ordinary JPG/PNG files because they do not prove location. "
        "If no UAV data exists for Jamanxim, use high-resolution satellite imagery and label it as satellite evidence."
    )
    uav_file = st.file_uploader("UAV orthomosaic GeoTIFF", type=["tif", "tiff"], key="uav-file")
    uav_date = st.text_input("UAV acquisition date (optional)", key="uav-date", placeholder="YYYY-MM-DD")
    uav_bbox = st.text_input(
        "Alert bbox as JSON (optional)", key="uav-bbox",
        placeholder="[-56.1, -7.2, -56.0, -7.1]",
    )
    if st.button("Validate UAV evidence", key="uav-run", use_container_width=True):
        if not uav_file:
            st.error("Choose a georeferenced GeoTIFF first.")
        else:
            try:
                with st.spinner("Checking CRS, footprint, resolution, and alert overlap…"):
                    response = requests.post(
                        f"{API}/upload-drone-orthomosaic",
                        files={"image_file": (uav_file.name, uav_file.getvalue(), "image/tiff")},
                        data={"alert_bbox": uav_bbox or "", "acquisition_date": uav_date or ""},
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    st.session_state["uav_result"] = response.json()
            except Exception as exc:
                show_api_error(exc)
    if st.session_state.get("uav_result"):
        uav_result = st.session_state["uav_result"]
        st.success(f"UAV evidence accepted for review: {uav_result.get('filename')}")
        st.json({k: v for k, v in uav_result.items() if k != "map_url"})
        if uav_result.get("map_url"):
            st.image(requests.get(f"{API}{uav_result['map_url']}", timeout=30).content, use_container_width=True)

    st.divider()
    st.subheader("Train an auditable activity model")
    st.caption(
        "The model is a Random Forest trained only from your labelled JSON rows. "
        "No accuracy is claimed until you upload labels and receive held-out validation metrics."
    )
    try:
        ml_state = api_get("/ml/status")
        if ml_state.get("trained"):
            st.success(f"Model trained · validation macro-F1: {ml_state.get('validation', {}).get('macro_f1')}")
        else:
            st.info("No trained model yet. Required features: " + ", ".join(ml_state.get("features", [])))
    except Exception as exc:
        show_api_error(exc)
    labels_file = st.file_uploader("Labelled feature JSON", type=["json"], key="labels-file")
    if st.button("Train and validate ML model", key="train-ml", use_container_width=True):
        if not labels_file:
            st.error("Upload labelled JSON rows first.")
        else:
            try:
                response = requests.post(
                    f"{API}/train-ml",
                    files={"labels_file": (labels_file.name, labels_file.getvalue(), "application/json")},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                st.session_state["ml_train_result"] = response.json()
            except Exception as exc:
                show_api_error(exc)
    if st.session_state.get("ml_train_result"):
        st.json(st.session_state["ml_train_result"].get("metrics", st.session_state["ml_train_result"]))

with history_tab:
    st.subheader("Saved analysis history")
    st.write("Only analyses actually run through this prototype appear here; no fictional alert examples are inserted.")
    if st.button("Refresh history", key="refresh-history"):
        st.session_state["refresh_history"] = True
    try:
        history_data = api_get("/history")
        entries = history_data.get("items", [])
        if not entries:
            st.info("No results saved yet. Run a satellite analysis or manual image test first.")
        else:
            for item in entries[:20]:
                heading = (
                    f"{item.get('created_at', '')[:19]} · {item.get('mode')} · "
                    f"{item.get('status')} · {item.get('job_id')}"
                )
                with st.expander(heading):
                    if item.get("mode") == "sentinel2":
                        st.write(
                            f"{item.get('before_date')} → {item.get('after_date')} | "
                            f"candidate area: {item.get('candidate_area_hectares')} ha | "
                            f"usable pixels: {item.get('valid_pixel_coverage_percent')}%"
                        )
                        st.write(item.get("explanation"))
                    else:
                        st.write(item.get("explanation"))
                    if item.get("map_url"):
                        try:
                            saved = requests.get(f"{API}{item['map_url']}", timeout=30)
                            saved.raise_for_status()
                            st.image(saved.content, use_container_width=True)
                        except requests.RequestException:
                            st.caption("The saved preview file is no longer available.")
    except Exception as exc:
        show_api_error(exc)

with monitor_tab:
    st.subheader("Scheduled Sentinel catalog checks")
    st.write(
        "This checks the real catalog on a schedule and runs the same analysis. Sentinel-2 is not a live video feed: "
        "a new check may find no new acquisition, and clouds can leave too few usable pixels. "
        "The schedule works only while the backend and this computer stay on."
    )
    interval = st.selectbox("Check interval", [6, 12, 24, 48], index=2, format_func=lambda x: f"Every {x} hours")
    monitor_payload = dict(request_payload)
    monitor_payload["interval_hours"] = int(interval)
    start_col, stop_col, status_col = st.columns(3)
    with start_col:
        if st.button("Start scheduled checks", type="primary", use_container_width=True):
            try:
                response = requests.post(f"{API}/monitor/start", json=monitor_payload, timeout=30)
                response.raise_for_status()
                st.success(response.json().get("message", "Scheduled checks started."))
            except Exception as exc:
                show_api_error(exc)
    with stop_col:
        if st.button("Stop scheduled checks", use_container_width=True):
            try:
                response = requests.post(f"{API}/monitor/stop", timeout=30)
                response.raise_for_status()
                st.info(response.json().get("message", "Stop requested."))
            except Exception as exc:
                show_api_error(exc)
    with status_col:
        if st.button("Refresh monitor status", use_container_width=True):
            st.session_state["refresh_monitor"] = True
    try:
        monitor = api_get("/monitor/status")
        st.write(f"**Status:** {monitor.get('status')} · **Checks completed:** {monitor.get('checks')}")
        st.write(f"**Last check:** {monitor.get('last_check') or 'Not yet checked'}")
        st.info(monitor.get("last_message", "No monitor activity yet."))
    except Exception as exc:
        show_api_error(exc)
