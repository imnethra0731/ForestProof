"""ForestProof prototype backend.

This prototype searches real Sentinel-2 L2A scenes in Earth Search, downloads
small COG windows, masks poor pixels using the SCL layer, calculates an NDVI
vegetation-decrease candidate, and intersects the candidate with ICMBio's
published boundary for Floresta Nacional do Jamanxim.

It does NOT decide whether an activity is illegal or classify construction,
mining, or roads. The manual RGB comparison is a separate visual test and is
not geospatial analysis.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from io import BytesIO
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform as transform_coords, transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping, shape
from ml_service import model_status, predict as ml_predict, train as ml_train


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "forestproof_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
HISTORY_FILE = OUTPUT_DIR / "history.json"
LABELS_FILE = OUTPUT_DIR / "labels.json"

STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"
ICMBIO_WFS_URL = "https://geoservicos.inde.gov.br/geoserver/ICMBio/ows"
ICMBIO_LAYER = "ICMBio:limiteucsfederais_a"
ICMBIO_FILTER = "nomeuc ILIKE '%FLORESTA NACIONAL DO JAMANXIM%'"
PRODES_WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"
PRODES_LAYER = "prodes-legal-amz:yearly_deforestation"
JAMANXIM_DEMO_PRODES_UUID = "d75dfa42-a812-40d1-a1d2-9df6320d85ba"
MAX_PIXELS = 3_000_000
HISTORY_LOCK = threading.Lock()
BOUNDARY_CACHE: dict[str, Any] = {"loaded_at": 0.0, "feature": None, "geometry": None}

app = FastAPI(
    title="ForestProof",
    description=(
        "Hackathon prototype for evidence-based forest-change review. "
        "Alerts indicate candidate vegetation change, not proof of illegal activity."
    ),
    version="1.0.0",
)


class SentinelAnalysisRequest(BaseModel):
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="West, south, east, north in longitude/latitude degrees.",
    )
    start_date: date
    end_date: date
    before_start_date: date | None = None
    before_end_date: date | None = None
    after_start_date: date | None = None
    after_end_date: date | None = None
    prodes_reference_uuid: str | None = None
    cloud_limit: float = Field(default=40.0, ge=0.0, le=100.0)
    min_gap_days: int = Field(default=30, ge=5, le=365)
    minimum_alert_area_ha: float = Field(default=0.5, ge=0.0, le=10000.0)
    interval_hours: int = Field(default=24, ge=1, le=168)


monitor_state: dict[str, Any] = {
    "running": False,
    "status": "Stopped",
    "interval_hours": None,
    "last_check": None,
    "last_message": "Scheduled checks are stopped.",
    "checks": 0,
}
monitor_lock = threading.Lock()
monitor_stop = threading.Event()
monitor_thread: threading.Thread | None = None


def _http_get_json(url: str, *, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach the public data service: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The public data service returned invalid JSON.") from exc


def get_jamanxim_boundary() -> tuple[dict[str, Any], Any]:
    """Get the official ICMBio polygon through its public WFS service."""
    now = time.time()
    cached_feature = BOUNDARY_CACHE.get("feature")
    if cached_feature is not None and now - float(BOUNDARY_CACHE.get("loaded_at", 0)) < 6 * 60 * 60:
        return cached_feature, BOUNDARY_CACHE["geometry"]

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": ICMBIO_LAYER,
        "outputFormat": "application/json",
        "CQL_FILTER": ICMBIO_FILTER,
        "count": 20,
        "srsName": "EPSG:4326",
    }
    collection = _http_get_json(ICMBIO_WFS_URL, params=params)
    features = collection.get("features", [])
    feature = next(
        (
            item for item in features
            if "FLORESTA NACIONAL DO JAMANXIM"
            in str(item.get("properties", {}).get("nomeuc", "")).upper()
        ),
        None,
    )
    if not feature or not feature.get("geometry"):
        raise HTTPException(
            status_code=502,
            detail="ICMBio did not return the Jamanxim National Forest boundary. Try again later.",
        )
    try:
        geom = shape(feature["geometry"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not read the official boundary geometry.") from exc

    BOUNDARY_CACHE.update({"loaded_at": now, "feature": feature, "geometry": geom})
    return feature, geom


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ForestProof backend is running.",
        "important": "Candidate vegetation change is not proof of illegal activity.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/area/jamanxim")
def area_jamanxim() -> dict[str, Any]:
    feature, geom = get_jamanxim_boundary()
    point = geom.representative_point()
    west, south, east, north = geom.bounds
    half_width = 0.04
    suggested_bbox = [
        max(west, point.x - half_width),
        max(south, point.y - half_width),
        min(east, point.x + half_width),
        min(north, point.y + half_width),
    ]
    properties = feature.get("properties", {})
    return {
        "name": properties.get("nomeuc", "Floresta Nacional do Jamanxim"),
        "official_area_ha": properties.get("areahaalb"),
        "boundary_geojson": feature,
        "bounds": [west, south, east, north],
        "suggested_bbox": suggested_bbox,
        "suggested_center": {"longitude": point.x, "latitude": point.y},
        "boundary_source": "ICMBio WFS: Limites das Unidades de Conservação Federais",
        "boundary_source_url": "https://geoservicos.inde.gov.br/geoserver/ICMBio/ows",
        "note": "Suggested box is only a small starting area. Check it on the map before analyzing.",
    }


@app.get("/demo-case/jamanxim-prodes-2025")
def jamanxim_prodes_demo_case() -> dict[str, Any]:
    """Fetch one real historical INPE PRODES feature used as demo context."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": PRODES_LAYER,
        "outputFormat": "application/json",
        "count": 5,
        "CQL_FILTER": f"uuid='{JAMANXIM_DEMO_PRODES_UUID}'",
    }
    collection = _http_get_json(PRODES_WFS_URL, params=params, timeout=90)
    feature = next(
        (item for item in collection.get("features", [])
         if item.get("properties", {}).get("uuid") == JAMANXIM_DEMO_PRODES_UUID),
        None,
    )
    if not feature or not feature.get("geometry"):
        raise HTTPException(status_code=502, detail="The selected INPE PRODES reference feature could not be retrieved.")
    boundary_feature, boundary = get_jamanxim_boundary()
    geometry = shape(feature["geometry"])
    if not boundary.intersects(geometry):
        raise HTTPException(status_code=502, detail="The INPE reference feature no longer intersects the selected reserve boundary.")
    point = geometry.representative_point()
    properties = feature.get("properties", {})
    west, south, east, north = geometry.bounds
    return {
        "label": "INPE PRODES 2025 historical reference feature",
        "feature": feature,
        "uuid": properties.get("uuid"),
        "year": properties.get("year"),
        "image_date": properties.get("image_date"),
        "sub_class": properties.get("sub_class"),
        "area_hectares": round(float(properties.get("area_km") or 0) * 100.0, 2),
        "source": "INPE PRODES, TerraBrasilis yearly deforestation layer",
        "source_url": "https://terrabrasilis.dpi.inpe.br/en/home-page/",
        "reference_point": {"latitude": point.y, "longitude": point.x},
        "suggested_bbox": [point.x - 0.04, point.y - 0.04, point.x + 0.04, point.y + 0.04],
        "boundary_name": boundary_feature.get("properties", {}).get("nomeuc"),
        "legal_status_note": "A PRODES class label and protected-area overlap are not, by themselves, a legal determination.",
    }


def _validate_bbox(request_data: SentinelAnalysisRequest, boundary: Any) -> tuple[float, float, float, float]:
    if len(request_data.bbox) != 4:
        raise HTTPException(status_code=400, detail="Enter four coordinates: west, south, east, north.")
    west, south, east, north = [float(x) for x in request_data.bbox]
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise HTTPException(
            status_code=400,
            detail="Invalid bounding box. Use west < east and south < north in longitude/latitude degrees.",
        )
    if (east - west) > 0.25 or (north - south) > 0.25:
        raise HTTPException(
            status_code=400,
            detail="For this demo, keep the selected map box at or below 0.25 degrees in each direction.",
        )
    if not boundary.intersects(box(west, south, east, north)):
        raise HTTPException(
            status_code=400,
            detail="The selected box does not overlap the official Jamanxim National Forest boundary.",
        )
    if request_data.start_date >= request_data.end_date:
        raise HTTPException(status_code=400, detail="Start date must be earlier than end date.")
    if (request_data.end_date - request_data.start_date).days < request_data.min_gap_days:
        raise HTTPException(
            status_code=400,
            detail="Choose a date range longer than the minimum before/after gap.",
        )
    return west, south, east, north


def _stac_items(bbox: tuple[float, float, float, float], request_data: SentinelAnalysisRequest) -> list[dict[str, Any]]:
    payload = {
        "collections": [STAC_COLLECTION],
        "bbox": list(bbox),
        "datetime": (
            f"{request_data.start_date.isoformat()}T00:00:00Z/"
            f"{request_data.end_date.isoformat()}T23:59:59Z"
        ),
        "limit": 300,
        "query": {"eo:cloud_cover": {"lt": request_data.cloud_limit}},
    }
    try:
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("features", [])
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not search the Sentinel-2 catalog: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The Sentinel-2 catalog returned invalid JSON.") from exc


def _stac_items_window(
    bbox: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    cloud_limit: float,
) -> list[dict[str, Any]]:
    """Search one user-selected date window independently."""
    payload = {
        "collections": [STAC_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
        "limit": 300,
        "query": {"eo:cloud_cover": {"lt": cloud_limit}},
    }
    try:
        response = requests.post(STAC_SEARCH_URL, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("features", [])
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not search the Sentinel-2 catalog: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The Sentinel-2 catalog returned invalid JSON.") from exc


def _item_tile(item: dict[str, Any]) -> str | None:
    props = item.get("properties", {})
    for key in ("s2:mgrs_tile", "mgrs:tile", "grid:code"):
        value = props.get(key)
        if value:
            return str(value)
    match = re.search(r"_(\d{2}[A-Z]{3})_", str(item.get("id", "")))
    return match.group(1) if match else None


def _item_date(item: dict[str, Any]) -> date:
    value = item.get("properties", {}).get("datetime", "")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except Exception:
        return date.min


def _select_pair(items: list[dict[str, Any]], minimum_gap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    items = [
        item for item in items
        if item.get("assets", {}).get("red")
        and item.get("assets", {}).get("nir")
        and item.get("assets", {}).get("scl")
    ]
    items.sort(key=lambda item: _item_date(item), reverse=True)
    for after in items:
        after_date = _item_date(after)
        after_tile = _item_tile(after)
        earlier = [
            item for item in items
            if item is not after
            and _item_date(item) <= after_date - timedelta(days=minimum_gap)
            and (not after_tile or _item_tile(item) == after_tile)
        ]
        if earlier:
            before = max(earlier, key=lambda item: _item_date(item))
            return before, after
    raise HTTPException(
        status_code=404,
        detail=(
            "No same-tile Sentinel-2 date pair was found in this period. "
            "Try a longer date range, a higher scene cloud limit, or a slightly different box."
        ),
    )


def _select_pair_from_windows(
    before_items: list[dict[str, Any]],
    after_items: list[dict[str, Any]],
    minimum_gap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose a low-cloud same-tile pair from two user-selected periods."""
    def valid(item: dict[str, Any]) -> bool:
        assets = item.get("assets", {})
        return all(assets.get(key) for key in ("red", "nir", "scl"))

    candidates = []
    for before in before_items:
        if not valid(before):
            continue
        for after in after_items:
            if not valid(after) or _item_tile(before) != _item_tile(after):
                continue
            gap = (_item_date(after) - _item_date(before)).days
            if gap < minimum_gap:
                continue
            before_cloud = float(before.get("properties", {}).get("eo:cloud_cover", 100.0))
            after_cloud = float(after.get("properties", {}).get("eo:cloud_cover", 100.0))
            candidates.append((before_cloud + after_cloud, max(before_cloud, after_cloud), before, after))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                "No same-tile Sentinel-2 pair was found inside both selected date periods. "
                "Widen either period, raise the cloud limit, or check that the periods are far enough apart."
            ),
        )
    # Scene-wide cloud metadata is only a proxy. Local usable coverage is still
    # calculated from the actual SCL bands after the pair is chosen.
    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    return candidates[0][2], candidates[0][3]


def _asset_href(item: dict[str, Any], key: str) -> str:
    asset = item.get("assets", {}).get(key)
    href = asset.get("href") if asset else None
    if not href:
        raise HTTPException(status_code=502, detail=f"Sentinel-2 scene is missing the {key} data asset.")
    return href


def _target_grid(after_item: dict[str, Any], bbox: tuple[float, float, float, float]):
    href = _asset_href(after_item, "red")
    try:
        with rasterio.open(href) as src:
            if not src.crs:
                raise HTTPException(status_code=502, detail="Satellite image has no map coordinate reference system.")
            projected_bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            window = from_bounds(*projected_bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            width, height = int(window.width), int(window.height)
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="The selected map box is outside this satellite tile.")
            if width * height > MAX_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="The requested image area is too large for this demo. Reduce the map box and try again.",
                )
            return src.crs, src.window_transform(window), width, height
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not open a Sentinel-2 Cloud-Optimized GeoTIFF. "
                "Check internet access and try another date or area. "
                f"Technical detail: {exc}"
            ),
        ) from exc


def _read_asset_to_grid(
    href: str,
    out_shape: tuple[int, int],
    out_transform: Any,
    out_crs: Any,
    resampling: Resampling,
) -> np.ndarray:
    height, width = out_shape
    out = np.zeros((height, width), dtype=np.float32)
    try:
        with rasterio.open(href) as src:
            src_bounds = transform_bounds(out_crs, src.crs, *rasterio.transform.array_bounds(height, width, out_transform), densify_pts=21)
            src_window = from_bounds(*src_bounds, transform=src.transform)
            src_window = src_window.round_offsets().round_lengths()
            src_window = src_window.intersection(Window(0, 0, src.width, src.height))
            if src_window.width <= 0 or src_window.height <= 0:
                return out
            src_data = src.read(1, window=src_window)
            src_transform = src.window_transform(src_window)
            nodata = src.nodata
            reproject(
                source=src_data,
                destination=out,
                src_transform=src_transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=out_transform,
                dst_crs=out_crs,
                dst_nodata=0,
                resampling=resampling,
                init_dest_nodata=True,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read a Sentinel-2 image band: {exc}") from exc
    return out


def _read_scene(item: dict[str, Any], shape_hw: tuple[int, int], transform: Any, crs: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in ("red", "green", "blue", "nir"):
        result[key] = _read_asset_to_grid(
            _asset_href(item, key), shape_hw, transform, crs, Resampling.bilinear
        )
    result["scl"] = _read_asset_to_grid(
        _asset_href(item, "scl"), shape_hw, transform, crs, Resampling.nearest
    ).astype(np.uint8)
    return result


def _stretch_rgb(scene: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    bands = [scene["red"], scene["green"], scene["blue"]]
    channels = []
    for band in bands:
        values = band[valid & (band > 0)]
        if values.size < 20:
            low, high = 0.0, 3000.0
        else:
            low, high = np.percentile(values, [2, 98])
            if high <= low:
                high = low + 1.0
        channel = np.clip((band - low) / (high - low), 0, 1)
        channels.append((channel * 255).astype(np.uint8))
    return np.dstack(channels)


def _make_satellite_preview(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    loss_mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    before_date: str,
    after_date: str,
) -> Image.Image:
    panel_w, panel_h, title_h = 480, 360, 58
    canvas = Image.new("RGB", (panel_w * 3, panel_h + title_h + 45), "#101820")
    draw = ImageDraw.Draw(canvas)
    labels = [
        f"Before: {before_date}",
        f"After: {after_date}",
        "Candidate NDVI decrease (red)",
    ]
    panels = [before_rgb, after_rgb, after_rgb.copy()]
    overlay = panels[2].copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[loss_mask] = (0.35 * overlay[loss_mask] + 0.65 * red[loss_mask]).astype(np.uint8)
    panels[2] = overlay
    for i, (label, array) in enumerate(zip(labels, panels)):
        draw.text((i * panel_w + 12, 10), label, fill="white")
        image = Image.fromarray(array, mode="RGB").resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas.paste(image, (i * panel_w, title_h))
    west, south, east, north = bbox
    draw.text(
        (12, panel_h + title_h + 12),
        f"Actual cropped Sentinel-2 image window | W,S,E,N: {west:.5f}, {south:.5f}, {east:.5f}, {north:.5f} | 10 m bands",
        fill="#cbd5e1",
    )
    return canvas


def _save_map(image: Image.Image, job_id: str) -> None:
    image.save(OUTPUT_DIR / f"{job_id}.png", format="PNG")


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_history_entry(entry: dict[str, Any]) -> None:
    with HISTORY_LOCK:
        history = _load_history()
        history.insert(0, entry)
        HISTORY_FILE.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")


def _item_self_url(item: dict[str, Any]) -> str | None:
    for link in item.get("links", []):
        if link.get("rel") == "self":
            return link.get("href")
    return None


def _centroid_coordinates(mask: np.ndarray, transform: Any, crs: Any) -> dict[str, float] | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    row = float(rows.mean())
    col = float(cols.mean())
    x, y = transform * (col + 0.5, row + 0.5)
    longitude, latitude = transform_coords(crs, "EPSG:4326", [x], [y])
    return {"latitude": round(float(latitude[0]), 6), "longitude": round(float(longitude[0]), 6)}


def _run_satellite_analysis(request_data: SentinelAnalysisRequest) -> dict[str, Any]:
    boundary_feature, boundary = get_jamanxim_boundary()
    bbox = _validate_bbox(request_data, boundary)
    reference_case = None
    reference_geometry = None
    if request_data.prodes_reference_uuid:
        if request_data.prodes_reference_uuid != JAMANXIM_DEMO_PRODES_UUID:
            raise HTTPException(status_code=400, detail="Unknown PRODES reference case. Load the built-in Jamanxim demo case.")
        reference_case = jamanxim_prodes_demo_case()
        reference_geometry = shape(reference_case["feature"]["geometry"])
        if not reference_geometry.intersects(box(*bbox)):
            raise HTTPException(status_code=400, detail="The selected analysis box does not include the PRODES reference polygon.")
    custom_dates = [
        request_data.before_start_date,
        request_data.before_end_date,
        request_data.after_start_date,
        request_data.after_end_date,
    ]
    if any(value is not None for value in custom_dates) and not all(value is not None for value in custom_dates):
        raise HTTPException(status_code=400, detail="Fill in all four before/after date-window fields.")
    if all(value is not None for value in custom_dates):
        before_start = request_data.before_start_date
        before_end = request_data.before_end_date
        after_start = request_data.after_start_date
        after_end = request_data.after_end_date
        if before_start > before_end or after_start > after_end:
            raise HTTPException(status_code=400, detail="Each date window's start must be on or before its end.")
        if after_end <= before_start:
            raise HTTPException(status_code=400, detail="The after period must be later than the before period.")
        before_items = _stac_items_window(bbox, before_start, before_end, request_data.cloud_limit)
        after_items = _stac_items_window(bbox, after_start, after_end, request_data.cloud_limit)
        if not before_items or not after_items:
            raise HTTPException(
                status_code=404,
                detail="No scenes found in one of the selected periods. Widen that period or raise the cloud limit.",
            )
        before_item, after_item = _select_pair_from_windows(
            before_items, after_items, request_data.min_gap_days
        )
    else:
        features = _stac_items(bbox, request_data)
        if not features:
            raise HTTPException(
                status_code=404,
                detail="No Sentinel-2 scenes were found for this box and date range. Widen the dates or move the box.",
            )
        before_item, after_item = _select_pair(features, request_data.min_gap_days)
    # A scheduled catalog check may see the same latest pair many times.
    # Return the saved analysis instead of adding duplicate history entries.
    for saved in _load_history():
        if (
            saved.get("mode") == "sentinel2"
            and saved.get("before_scene_id") == before_item.get("id")
            and saved.get("after_scene_id") == after_item.get("id")
            and saved.get("prodes_reference_uuid") == request_data.prodes_reference_uuid
        ):
            return saved
    target_crs, target_transform, width, height = _target_grid(after_item, bbox)
    shape_hw = (height, width)

    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
    ):
        before = _read_scene(before_item, shape_hw, target_transform, target_crs)
        after = _read_scene(after_item, shape_hw, target_transform, target_crs)

    bbox_geom = box(*bbox)
    inside_geometry = boundary.intersection(bbox_geom)
    projected_geometry = transform_geom("EPSG:4326", target_crs, mapping(inside_geometry))
    boundary_mask = geometry_mask(
        [projected_geometry], out_shape=shape_hw, transform=target_transform, invert=True
    )
    reference_mask = np.zeros(shape_hw, dtype=bool)
    if reference_geometry is not None:
        projected_reference = transform_geom("EPSG:4326", target_crs, mapping(reference_geometry))
        reference_mask = geometry_mask(
            [projected_reference], out_shape=shape_hw, transform=target_transform, invert=True
        ) & boundary_mask

    # Sentinel-2 L2A SCL classes: mask no-data, saturated, cloud shadow,
    # unclassified/low-probability cloud, medium/high cloud, cirrus and snow.
    invalid_scl = np.array([0, 1, 3, 7, 8, 9, 10, 11], dtype=np.uint8)
    before_valid = (~np.isin(before["scl"], invalid_scl)) & (before["red"] > 0) & (before["nir"] > 0)
    after_valid = (~np.isin(after["scl"], invalid_scl)) & (after["red"] > 0) & (after["nir"] > 0)
    valid = before_valid & after_valid & boundary_mask
    boundary_pixel_count = int(boundary_mask.sum())
    valid_coverage = (float(valid.sum()) / boundary_pixel_count * 100.0) if boundary_pixel_count else 0.0
    enough_quality = valid_coverage >= 50.0

    red_before = before["red"] * 0.0001
    nir_before = before["nir"] * 0.0001
    red_after = after["red"] * 0.0001
    nir_after = after["nir"] * 0.0001
    ndvi_before = np.divide(
        nir_before - red_before,
        nir_before + red_before,
        out=np.zeros_like(nir_before, dtype=np.float32),
        where=(nir_before + red_before) > 0,
    )
    ndvi_after = np.divide(
        nir_after - red_after,
        nir_after + red_after,
        out=np.zeros_like(nir_after, dtype=np.float32),
        where=(nir_after + red_after) > 0,
    )

    baseline_forest = valid & (ndvi_before >= 0.50)
    raw_loss = (
        baseline_forest
        & (ndvi_after < 0.50)
        & ((ndvi_before - ndvi_after) >= 0.20)
    )
    baseline_pixels = int(baseline_forest.sum())
    loss_pixels = int(raw_loss.sum())
    loss_percent = (loss_pixels / baseline_pixels * 100.0) if baseline_pixels else 0.0
    pixel_area_m2 = abs(
        target_transform.a * target_transform.e - target_transform.b * target_transform.d
    )
    loss_area_ha = loss_pixels * pixel_area_m2 / 10000.0
    location = _centroid_coordinates(raw_loss, target_transform, target_crs)
    if not enough_quality:
        # Do not show a partial-area calculation as an alert when most of the
        # protected-area box has no usable paired observation.
        raw_loss = np.zeros_like(raw_loss, dtype=bool)
        baseline_pixels = None
        loss_pixels = None
        loss_percent = None
        loss_area_ha = None
        location = None
        prodes_overlap_percent = None
    elif reference_case:
        reference_pixels = int(reference_mask.sum())
        prodes_overlap_percent = (
            round(int((raw_loss & reference_mask).sum()) / reference_pixels * 100.0, 2)
            if reference_pixels else 0.0
        )
    else:
        prodes_overlap_percent = None

    before_visual_valid = before_valid & boundary_mask
    after_visual_valid = after_valid & boundary_mask
    before_rgb = _stretch_rgb(before, before_visual_valid)
    after_rgb = _stretch_rgb(after, after_visual_valid)
    image = _make_satellite_preview(
        before_rgb,
        after_rgb,
        raw_loss,
        bbox,
        _item_date(before_item).isoformat(),
        _item_date(after_item).isoformat(),
    )

    job_id = uuid.uuid4().hex[:12]
    _save_map(image, job_id)
    if not enough_quality:
        status = "insufficient_image_quality"
        explanation = (
            f"Only {valid_coverage:.1f}% of pixels inside the protected boundary were usable in both images. "
            "At least 50% usable paired coverage is required by this prototype. "
            "Clouds, shadows, or missing data make this comparison too weak for an alert."
        )
    elif baseline_pixels == 0:
        status = "insufficient_baseline"
        explanation = "No usable baseline forest-like pixels were found inside the selected protected-area box."
    elif loss_area_ha >= request_data.minimum_alert_area_ha:
        status = "review_recommended"
        explanation = (
            f"A candidate vegetation decrease was found in {loss_area_ha:.2f} ha of baseline forest-like pixels. "
            f"The estimate uses an NDVI drop of at least 0.20 and an after-NDVI below 0.50, with clouds/shadows "
            f"masked using the Sentinel-2 scene-classification layer. It overlaps the official protected boundary. "
            "This is not proof of deforestation, an illegal act, or a specific activity."
        )
    else:
        status = "no_candidate_above_area_threshold"
        explanation = (
            f"The selected rule found {loss_area_ha:.2f} ha of candidate vegetation decrease, below the "
            f"{request_data.minimum_alert_area_ha:.2f} ha review threshold. A two-date comparison cannot verify persistence."
        )

    after_props = after_item.get("properties", {})
    before_props = before_item.get("properties", {})
    if not enough_quality:
        activity_label = "Not assessed; insufficient usable imagery"
    elif loss_area_ha is not None and loss_area_ha >= request_data.minimum_alert_area_ha:
        activity_label = "Possible vegetation decrease; activity type not classified"
    else:
        activity_label = "No candidate above the selected review threshold"
    severity = (
        "high" if status == "review_recommended" and (loss_area_ha or 0) >= 5.0
        else "medium" if status == "review_recommended"
        else "low"
    )
    result: dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "sentinel2",
        "status": status,
        "severity": severity,
        "activity_label": activity_label,
        "prodes_reference_uuid": request_data.prodes_reference_uuid,
        "before_date": _item_date(before_item).isoformat(),
        "after_date": _item_date(after_item).isoformat(),
        "before_scene_id": before_item.get("id"),
        "after_scene_id": after_item.get("id"),
        "before_scene_cloud_cover_percent": round(float(before_props.get("eo:cloud_cover", 0)), 2),
        "after_scene_cloud_cover_percent": round(float(after_props.get("eo:cloud_cover", 0)), 2),
        "valid_pixel_coverage_percent": round(valid_coverage, 2),
        "baseline_forest_like_pixels": baseline_pixels,
        "mean_ndvi_before": round(float(np.mean(ndvi_before[valid])) if valid.any() else 0.0, 4),
        "mean_ndvi_after": round(float(np.mean(ndvi_after[valid])) if valid.any() else 0.0, 4),
        "mean_ndvi_drop": round(float(np.mean((ndvi_before - ndvi_after)[valid])) if valid.any() else 0.0, 4),
        "mean_red_before": round(float(np.mean(red_before[valid])) if valid.any() else 0.0, 6),
        "mean_nir_before": round(float(np.mean(nir_before[valid])) if valid.any() else 0.0, 6),
        "mean_red_after": round(float(np.mean(red_after[valid])) if valid.any() else 0.0, 6),
        "mean_nir_after": round(float(np.mean(nir_after[valid])) if valid.any() else 0.0, 6),
        "candidate_loss_pixels": loss_pixels,
        "candidate_vegetation_decrease_percent_of_baseline": round(loss_percent, 2) if loss_percent is not None else None,
        "candidate_area_hectares": round(loss_area_ha, 3) if loss_area_ha is not None else None,
        "minimum_alert_area_hectares": request_data.minimum_alert_area_ha,
        "ndvi_drop_threshold": 0.20,
        "baseline_forest_like_ndvi_threshold": 0.50,
        "protected_boundary_overlap": bool(loss_pixels > 0) if loss_pixels is not None else None,
        "candidate_centroid": location,
        "prodes_reference": ({
            "uuid": reference_case["uuid"],
            "year": reference_case["year"],
            "image_date": reference_case["image_date"],
            "class_label": reference_case["sub_class"],
            "area_hectares": reference_case["area_hectares"],
            "source": reference_case["source"],
            "source_url": reference_case["source_url"],
            "legal_status_note": reference_case["legal_status_note"],
            "feature": reference_case["feature"],
            "ndvi_candidate_overlap_percent": prodes_overlap_percent,
        } if reference_case else None),
        "bbox_wsen": [round(x, 6) for x in bbox],
        "image_resolution_meters": 10,
        "sensor": "Sentinel-2 L2A",
        "boundary_name": boundary_feature.get("properties", {}).get("nomeuc"),
        "boundary_source": "ICMBio WFS, Federal Conservation Unit Boundaries",
        "boundary_source_url": "https://geoservicos.inde.gov.br/geoserver/ICMBio/ows",
        "imagery_catalog": "Element 84 Earth Search STAC, Sentinel-2 L2A",
        "imagery_catalog_url": STAC_SEARCH_URL,
        "before_source_url": _item_self_url(before_item),
        "after_source_url": _item_self_url(after_item),
        "cloud_shadow_filter": "SCL classes 0, 1, 3, 7, 8, 9, 10 and 11 excluded",
        "persistence_check": "Not performed; only the selected before/after pair is analyzed.",
        "explanation": explanation,
        "map_url": f"/maps/{job_id}.png",
        "limitations": [
            "NDVI decrease is a candidate vegetation change, not proof of illegal activity.",
            "This prototype does not classify construction, roads, or mining.",
            "The cloud percentage in catalog metadata describes a scene, not only this selected box.",
            "The threshold values are prototype rules and need local validation.",
        ],
    }
    _save_history_entry(result)
    return result


@app.post("/analyze-sentinel")
def analyze_sentinel(request_data: SentinelAnalysisRequest) -> dict[str, Any]:
    return _run_satellite_analysis(request_data)


@app.post("/upload-manual")
async def upload_manual(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
) -> dict[str, Any]:
    """Optional manual RGB-pixel comparison; it does not provide geospatial results."""
    try:
        before_image = Image.open(BytesIO(await before_file.read())).convert("RGB")
        after_image = Image.open(BytesIO(await after_file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Upload two valid PNG or JPG image files.") from exc
    if before_image.size != after_image.size:
        raise HTTPException(
            status_code=400,
            detail=(
                "The manual test needs matching pixel dimensions. This endpoint does not stretch or align images. "
                f"Before is {before_image.width}x{before_image.height}; after is {after_image.width}x{after_image.height}."
            ),
        )
    if before_image.width * before_image.height > 12_000_000:
        raise HTTPException(status_code=400, detail="Images are too large for this manual test; use images under 12 megapixels.")

    before_array = np.asarray(before_image, dtype=np.int16)
    after_array = np.asarray(after_image, dtype=np.int16)
    difference = np.abs(after_array - before_array).mean(axis=2)
    changed_mask = difference > 30
    changed_percent = float(changed_mask.mean() * 100.0)
    overlay = np.asarray(after_image, dtype=np.uint8).copy()
    overlay[changed_mask] = (0.35 * overlay[changed_mask] + 0.65 * np.array([255, 30, 30])).astype(np.uint8)
    job_id = uuid.uuid4().hex[:12]
    preview = Image.fromarray(overlay, mode="RGB")
    _save_map(preview, job_id)
    result = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "manual_rgb_test",
        "status": "visual_comparison_complete",
        "image_width": before_image.width,
        "image_height": before_image.height,
        "changed_pixel_percent": round(changed_percent, 2),
        "activity_label": "Not classified",
        "coordinates": None,
        "vegetation_loss_percent": None,
        "protected_boundary_overlap": None,
        "explanation": (
            "This is a simple RGB pixel-difference test on user-uploaded images. "
            "It does not calculate NDVI, know the location, or identify deforestation."
        ),
        "map_url": f"/maps/{job_id}.png",
        "limitations": [
            "No geographic coordinates or map alignment are available for these ordinary image uploads.",
            "RGB pixel difference is not vegetation loss and may react to lighting, shadows, or image misalignment.",
        ],
    }
    _save_history_entry(result)
    return result


@app.get("/history")
def history() -> dict[str, Any]:
    return {"items": _load_history()}


@app.get("/maps/{job_id}.png")
def get_map(job_id: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(status_code=404, detail="Map not found.")
    path = OUTPUT_DIR / f"{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Map not found. Run an analysis first.")
    return FileResponse(path, media_type="image/png")






class TimeSeriesAnalysisRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    start_date: date
    end_date: date
    cloud_limit: float = Field(default=40.0, ge=0.0, le=100.0)
    minimum_observations: int = Field(default=4, ge=3, le=12)
    seasonal_months_only: bool = True


def _month_distance(month_a: int, month_b: int) -> int:
    """Circular month distance: December and January are adjacent."""
    delta = abs(int(month_a) - int(month_b))
    return min(delta, 12 - delta)


def _latest_valid_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((obs for obs in reversed(observations) if obs.get("valid")), None)


def _priority_from_evidence(latest_drop: float, persistence_count: int, total_observations: int, coverage: float, seasonal_risk: str) -> tuple[str, int]:
    """Deterministic triage score; not a probability of illegality."""
    drop_score = min(35, int(max(0.0, latest_drop) / 0.40 * 35))
    persistence_score = min(30, int((persistence_count / max(total_observations, 1)) * 30))
    coverage_score = 20 if coverage >= 80 else 12 if coverage >= 50 else 0
    seasonal_score = 15 if seasonal_risk == "low" else 7 if seasonal_risk == "medium" else 0
    score = max(0, min(100, drop_score + persistence_score + coverage_score + seasonal_score))
    return ("high" if score >= 70 else "medium" if score >= 40 else "low", score)


def _load_labels() -> list[dict[str, Any]]:
    if not LABELS_FILE.exists():
        return []
    try:
        data = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_label(label: dict[str, Any]) -> dict[str, Any]:
    with HISTORY_LOCK:
        labels = [item for item in _load_labels() if item.get("sample_id") != label.get("sample_id")]
        labels.insert(0, label)
        LABELS_FILE.write_text(json.dumps(labels[:5000], indent=2), encoding="utf-8")
    return label


class LabelSampleRequest(BaseModel):
    sample_id: str = Field(..., min_length=1, max_length=160)
    group: str = Field(..., min_length=1, max_length=160)
    label: str = Field(..., min_length=1, max_length=80)
    review_status: str = Field(default="accepted", pattern="^(accepted|rejected|uncertain)$")
    latitude: float | None = None
    longitude: float | None = None
    before_date: date | None = None
    after_date: date | None = None
    label_source: str = Field(default="user_review", max_length=300)
    reviewer_confidence: str = Field(default="medium", pattern="^(low|medium|high)$")
    notes: str = Field(default="", max_length=2000)
    features: dict[str, float] = Field(default_factory=dict)


@app.get("/labels")
def list_labels(label: str | None = None, review_status: str | None = None) -> dict[str, Any]:
    items = _load_labels()
    if label and label != "all":
        items = [item for item in items if item.get("label") == label]
    if review_status and review_status != "all":
        items = [item for item in items if item.get("review_status") == review_status]
    return {"items": items, "count": len(items)}


@app.get("/ml/readiness")
def ml_readiness() -> dict[str, Any]:
    accepted = [item for item in _load_labels() if item.get("review_status") == "accepted"]
    classes = sorted({str(item.get("label")) for item in accepted if item.get("label")})
    count = len(accepted)
    ready = count >= 10 and len(classes) >= 2
    return {
        "ready": ready, "accepted_count": count, "available_classes": classes,
        "minimum_accepted_samples": 10, "minimum_classes": 2,
        "message": "Ready for genuine-label validation." if ready else "ML training is optional and not ready: collect at least 10 accepted genuine samples across at least 2 classes.",
    }


@app.post("/labels")
def save_label(request_data: LabelSampleRequest) -> dict[str, Any]:
    if request_data.review_status == "accepted":
        required = {"ndvi_before", "ndvi_after", "ndvi_drop", "red_before", "nir_before", "red_after", "nir_after", "valid_coverage_percent", "candidate_area_hectares"}
        missing = sorted(required - set(request_data.features))
        if missing:
            raise HTTPException(status_code=400, detail=f"Accepted ML labels need feature values: {', '.join(missing)}")
    item = request_data.model_dump(mode="json")
    item["created_at"] = datetime.now(timezone.utc).isoformat()
    _save_label(item)
    return item


@app.post("/labels/from-analysis/{job_id}")
def create_label_from_analysis(job_id: str, group: str = "unassigned_site") -> dict[str, Any]:
    """Create an uncertain review sample from a real saved Sentinel-2 analysis."""
    entry = next((item for item in _load_history() if item.get("job_id") == job_id and item.get("mode") == "sentinel2"), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Sentinel-2 analysis job not found.")
    features = {
        "ndvi_before": float(entry.get("mean_ndvi_before", 0.0)),
        "ndvi_after": float(entry.get("mean_ndvi_after", 0.0)),
        "ndvi_drop": float(entry.get("mean_ndvi_drop", 0.0)),
        "red_before": float(entry.get("mean_red_before", 0.0)),
        "nir_before": float(entry.get("mean_nir_before", 0.0)),
        "red_after": float(entry.get("mean_red_after", 0.0)),
        "nir_after": float(entry.get("mean_nir_after", 0.0)),
        "valid_coverage_percent": float(entry.get("valid_pixel_coverage_percent", 0.0)),
        "candidate_area_hectares": float(entry.get("candidate_area_hectares") or 0.0),
    }
    location = entry.get("candidate_centroid") or {}
    return _save_label({
        "sample_id": f"analysis_{job_id}", "group": group, "label": "unknown",
        "review_status": "uncertain", "latitude": location.get("latitude"), "longitude": location.get("longitude"),
        "before_date": entry.get("before_date"), "after_date": entry.get("after_date"),
        "label_source": "real Sentinel-2 analysis; requires human review", "reviewer_confidence": "low",
        "notes": "Imported from a real saved satellite analysis. Review before accepting.",
        "severity": entry.get("severity", "not_assigned"),
        "possible_change": entry.get("activity_label", "candidate vegetation change"),
        "features": features, "source_job_id": job_id, "map_url": entry.get("map_url"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/labels/export")
def export_labels() -> dict[str, Any]:
    accepted = [item for item in _load_labels() if item.get("review_status") == "accepted"]
    return {"rows": [{**item.get("features", {}), "label": item.get("label"), "group": item.get("group")} for item in accepted], "accepted_count": len(accepted)}


@app.post("/train-ml-from-labels")
def train_ml_from_labels() -> dict[str, Any]:
    rows = export_labels()["rows"]
    readiness = ml_readiness()
    if not readiness["ready"]:
        raise HTTPException(status_code=400, detail=readiness["message"])
    try:
        metrics = ml_train(rows)
        return {"status": "trained", "metrics": metrics, "accepted_count": len(rows)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze-timeseries")
def analyze_timeseries(request_data: TimeSeriesAnalysisRequest) -> dict[str, Any]:
    """Analyze multiple acquisitions and require persistent change before escalation.

    This first implementation reports a protected-area time-series summary. It is
    deliberately conservative: it does not claim pixel-level segmentation from
    mean values and marks seasonality risk when the requested dates span months.
    """
    if request_data.start_date >= request_data.end_date:
        raise HTTPException(status_code=400, detail="Start date must be earlier than end date.")
    boundary_feature, boundary = get_jamanxim_boundary()
    request_base = SentinelAnalysisRequest(
        bbox=request_data.bbox, start_date=request_data.start_date,
        end_date=request_data.end_date, cloud_limit=request_data.cloud_limit,
    )
    bbox = _validate_bbox(request_base, boundary)
    items = _stac_items_window(bbox, request_data.start_date, request_data.end_date, request_data.cloud_limit)
    valid_items = [i for i in items if all(i.get("assets", {}).get(k) for k in ("red", "green", "blue", "nir", "scl"))]
    if request_data.seasonal_months_only:
        # Prefer one seasonal window around the median acquisition month.
        months = [_item_date(i).month for i in valid_items if _item_date(i) != date.min]
        if months:
            target_month = sorted(months)[len(months) // 2]
            seasonal = [i for i in valid_items if _month_distance(_item_date(i).month, target_month) <= 1]
            if len(seasonal) >= request_data.minimum_observations:
                valid_items = seasonal
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in valid_items:
        groups.setdefault(_item_tile(item) or "unknown", []).append(item)
    if not groups:
        raise HTTPException(status_code=404, detail="No usable multi-date Sentinel-2 scenes were found.")
    tile, tile_items = max(groups.items(), key=lambda pair: len(pair[1]))
    tile_items.sort(key=lambda item: (_item_date(item), float(item.get("properties", {}).get("eo:cloud_cover", 100))))
    if len(tile_items) < request_data.minimum_observations:
        raise HTTPException(status_code=404, detail=f"Only {len(tile_items)} usable observations found; need at least {request_data.minimum_observations}.")
    tile_items = tile_items[-12:]
    target_crs, target_transform, width, height = _target_grid(tile_items[-1], bbox)
    shape_hw = (height, width)
    projected_geometry = transform_geom("EPSG:4326", target_crs, mapping(boundary.intersection(box(*bbox))))
    boundary_mask = geometry_mask([projected_geometry], out_shape=shape_hw, transform=target_transform, invert=True)
    observations: list[dict[str, Any]] = []
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES"):
        for item in tile_items:
            scene = _read_scene(item, shape_hw, target_transform, target_crs)
            invalid_scl = np.array([0, 1, 3, 7, 8, 9, 10, 11], dtype=np.uint8)
            valid = (~np.isin(scene["scl"], invalid_scl)) & (scene["red"] > 0) & (scene["nir"] > 0) & boundary_mask
            red = scene["red"] * 0.0001
            nir = scene["nir"] * 0.0001
            ndvi = np.divide(nir-red, nir+red, out=np.zeros_like(red), where=(nir+red)>0)
            coverage = float(valid.sum() / max(boundary_mask.sum(), 1) * 100)
            observations.append({
                "date": _item_date(item).isoformat(),
                "scene_id": item.get("id"),
                "cloud_metadata_percent": round(float(item.get("properties", {}).get("eo:cloud_cover", 100)), 2),
                "usable_coverage_percent": round(coverage, 2),
                "mean_ndvi": round(float(np.mean(ndvi[valid])) if valid.any() else 0.0, 4),
                "valid": bool(coverage >= 50),
                "source_url": _item_self_url(item),
            })
    usable = [o for o in observations if o["valid"]]
    if len(usable) < 2:
        raise HTTPException(status_code=422, detail="Too few observations passed local cloud/shadow quality screening.")
    baseline = float(np.median([o["mean_ndvi"] for o in usable[:max(2, len(usable)//3)]]))
    for obs in observations:
        obs["ndvi_drop_from_baseline"] = round(max(0.0, baseline - obs["mean_ndvi"]) if obs["valid"] else 0.0, 4)
    persistent = [o for o in usable if o["ndvi_drop_from_baseline"] >= 0.15]
    persistence_count = len(persistent)
    latest_valid = _latest_valid_observation(observations)
    if latest_valid is None:
        raise HTTPException(status_code=422, detail="No valid latest observation remains after quality screening.")
    latest_drop = latest_valid["ndvi_drop_from_baseline"]
    observed_months = {date.fromisoformat(o["date"]).month for o in usable}
    seasonal_risk = "low" if len(observed_months) <= 2 else "medium"
    if latest_drop >= 0.25 and persistence_count >= 2:
        status = "persistent_change_review"
    elif latest_drop >= 0.15:
        status = "candidate_change"
    else:
        status = "no_persistent_change"
    priority_level, priority_score = _priority_from_evidence(
        latest_drop, persistence_count, len(usable),
        float(np.mean([o["usable_coverage_percent"] for o in usable])), seasonal_risk
    )
    result = {
        "job_id": uuid.uuid4().hex[:12], "mode": "sentinel2_timeseries", "status": status,
        "tile": tile, "bbox_wsen": list(bbox), "observations": observations,
        "baseline_mean_ndvi": round(baseline, 4), "latest_valid_observation_date": latest_valid["date"], "latest_ndvi_drop": round(latest_drop, 4),
        "persistence_observation_count": persistence_count, "seasonal_risk": seasonal_risk,
        "severity": "high" if persistence_count >= 3 and latest_drop >= 0.25 else "medium" if latest_drop >= 0.15 else "low",
        "inspection_priority": priority_level, "priority_score": priority_score,
        "explanation": "Multi-date trend summary. Persistent decrease is a review signal, not proof of a specific activity or illegality.",
        "boundary_name": boundary_feature.get("properties", {}).get("nomeuc"),
        "imagery_catalog_url": STAC_SEARCH_URL,
        "limitations": ["This endpoint summarizes mean NDVI and does not create a pixel-level change polygon.", "Seasonality and fire/flooding require additional labelled or external masks.", "Use independent labelled sites or dates for ML evaluation."],
    }
    _save_history_entry(result)
    return result


@app.get("/evidence-report/{job_id}")
def evidence_report(job_id: str) -> dict[str, Any]:
    """Return a portable evidence report for a saved analysis."""
    for entry in _load_history():
        if entry.get("job_id") == job_id:
            return {"report_type": "ForestProof evidence review", "generated_at": datetime.now(timezone.utc).isoformat(), "result": entry}
    raise HTTPException(status_code=404, detail="Analysis job not found.")


class MLPredictRequest(BaseModel):
    features: dict[str, float]


@app.get("/ml/status")
def ml_status() -> dict[str, Any]:
    """Return whether a labelled activity model is available and its validation record."""
    return model_status()


@app.post("/train-ml")
async def train_ml(labels_file: UploadFile = File(...)) -> dict[str, Any]:
    """Train an auditable Random Forest from a JSON list of labelled feature rows.

    Required fields are: label plus the feature names listed by /ml/status.
    """
    raw = await labels_file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Upload a JSON list, or an object with a 'rows' list.")
        metrics = ml_train(rows)
        return {"status": "trained", "metrics": metrics, "model": model_status()}
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-ml")
def predict_ml(request_data: MLPredictRequest) -> dict[str, Any]:
    try:
        return ml_predict(request_data.features)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/upload-drone-orthomosaic")
async def upload_drone_orthomosaic(
    image_file: UploadFile = File(...),
    alert_bbox: str | None = Form(default=None),
    acquisition_date: str | None = Form(default=None),
) -> dict[str, Any]:
    """Validate a georeferenced drone/UAV GeoTIFF and create a review preview.

    This endpoint deliberately rejects ordinary JPG/PNG uploads: a verification
    layer must carry a CRS and geographic bounds.
    """
    raw = await image_file.read()
    try:
        with MemoryFile(raw) as memfile:
            with memfile.open() as src:
                if not src.crs or not src.bounds:
                    raise ValueError("The UAV file must contain a CRS and geographic bounds.")
                if src.count < 3:
                    raise ValueError("The UAV orthomosaic must contain at least three bands (RGB).")
                bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                if alert_bbox:
                    try:
                        alert = [float(v) for v in json.loads(alert_bbox)]
                        if len(alert) != 4:
                            raise ValueError
                        if not box(*alert).intersects(box(*bounds4326)):
                            raise ValueError("The UAV footprint does not overlap the supplied alert bounding box.")
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(str(exc) or "alert_bbox must be JSON [west,south,east,north].") from exc
                # Read a bounded overview for a safe, quick evidence image.
                scale = max(src.width / 1400, src.height / 1000, 1)
                out_h, out_w = max(1, int(src.height / scale)), max(1, int(src.width / scale))
                data = src.read([1, 2, 3], out_shape=(3, out_h, out_w), masked=True)
                arr = np.ma.filled(data, 0).astype(np.float32)
                lo, hi = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
                rgb = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
                preview = Image.fromarray(np.moveaxis((rgb * 255).astype(np.uint8), 0, 2), mode="RGB")
                job_id = uuid.uuid4().hex[:12]
                _save_map(preview, job_id)
                result = {
                    "job_id": job_id,
                    "mode": "georeferenced_uav_verification",
                    "status": "accepted_for_review",
                    "filename": image_file.filename,
                    "acquisition_date": acquisition_date,
                    "crs": str(src.crs),
                    "band_count": src.count,
                    "width": src.width,
                    "height": src.height,
                    "resolution_meters": [abs(float(src.res[0])), abs(float(src.res[1]))],
                    "footprint_wsen": [round(float(v), 8) for v in bounds4326],
                    "alert_overlap_checked": bool(alert_bbox),
                    "map_url": f"/maps/{job_id}.png",
                    "evidence_note": "UAV evidence is location-linked, but it does not by itself prove activity legality or classification.",
                }
                _save_history_entry(result)
                return result
    except HTTPException:
        raise
    except (ValueError, rasterio.errors.RasterioIOError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid georeferenced UAV GeoTIFF: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the UAV orthomosaic: {exc}") from exc


def _monitor_loop(initial_request: SentinelAnalysisRequest) -> None:
    global monitor_state
    span_days = max(60, (initial_request.end_date - initial_request.start_date).days)
    interval_seconds = initial_request.interval_hours * 3600
    while not monitor_stop.is_set():
        today = date.today()
        rolling_payload = {
            **initial_request.model_dump(),
            "start_date": today - timedelta(days=span_days),
            "end_date": today,
            # Monitoring must discover new acquisitions instead of reusing a
            # one-off historical before/after demo window.
            "before_start_date": None,
            "before_end_date": None,
            "after_start_date": None,
            "after_end_date": None,
        }
        rolling_request = SentinelAnalysisRequest(**rolling_payload)
        try:
            result = _run_satellite_analysis(rolling_request)
            message = (
                f"Catalog check complete: {result['before_scene_id']} → {result['after_scene_id']}; "
                f"status: {result['status']}."
            )
        except Exception as exc:
            message = f"Catalog check did not complete: {getattr(exc, 'detail', str(exc))}"
        with monitor_lock:
            monitor_state["checks"] += 1
            monitor_state["last_check"] = datetime.now(timezone.utc).isoformat()
            monitor_state["last_message"] = message
            monitor_state["status"] = "Running" if not monitor_stop.is_set() else "Stopped"
        if monitor_stop.wait(interval_seconds):
            break
    with monitor_lock:
        monitor_state["running"] = False
        monitor_state["status"] = "Stopped"


@app.post("/monitor/start")
def start_monitor(request_data: SentinelAnalysisRequest) -> dict[str, Any]:
    global monitor_thread, monitor_state
    get_jamanxim_boundary()
    _validate_bbox(request_data, BOUNDARY_CACHE["geometry"])
    with monitor_lock:
        if monitor_state.get("running"):
            return {"message": "Scheduled catalog monitoring is already running.", "monitor": monitor_state}
        monitor_stop.clear()
        monitor_state.update({
            "running": True,
            "status": "Starting; first source check will run now",
            "interval_hours": request_data.interval_hours,
            "last_message": "Starting the real catalog search worker.",
        })
        monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(request_data,),
            name="forestproof-sentinel-monitor",
            daemon=True,
        )
        monitor_thread.start()
    return {
        "message": (
            f"Scheduled real catalog checks started every {request_data.interval_hours} hour(s). "
            "They run only while this backend process and computer remain on."
        ),
        "monitor": monitor_state,
    }


@app.post("/monitor/stop")
def stop_monitor() -> dict[str, Any]:
    global monitor_state
    monitor_stop.set()
    with monitor_lock:
        monitor_state["running"] = False
        monitor_state["status"] = "Stopping" if monitor_thread and monitor_thread.is_alive() else "Stopped"
        monitor_state["last_message"] = "Stop requested."
    return {"message": "Stop requested.", "monitor": monitor_state}


@app.get("/monitor/status")
def get_monitor_status() -> dict[str, Any]:
    with monitor_lock:
        return dict(monitor_state)
