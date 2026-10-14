"""
Wayfinder OpenStreetMap Router.
Queries the Open Source Routing Machine (OSRM) API for exact road geometries,
driving turn-by-turn paths, and live travel times on OpenStreetMap roads.
"""

from typing import List, Tuple, Dict, Any, Optional
import urllib.request
import json


OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving"


def get_road_route_from_osrm(coordinates: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    """
    Given a list of (lat, lng) tuples, calls OSRM driving router and returns:
    - road_points: list of [lat, lng] points snapped to OpenStreetMap roads
    - distance_km: total driving distance in kilometers
    - duration_min: estimated travel duration in minutes
    """
    if len(coordinates) < 2:
        return None

    # OSRM expects {lng},{lat};{lng},{lat}
    coord_strs = [f"{lng:.6f},{lat:.6f}" for lat, lng in coordinates]
    url = f"{OSRM_BASE_URL}/{';'.join(coord_strs)}?overview=full&geometries=geojson"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Wayfinder-RTL-Router/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                raw_coords = route["geometry"]["coordinates"]
                # Convert [lon, lat] -> [lat, lon] for Leaflet
                lat_lng_points = [[pt[1], pt[0]] for pt in raw_coords]
                return {
                    "source": "OpenStreetMap OSRM Routing Engine",
                    "road_points": lat_lng_points,
                    "distance_km": round(route["distance"] / 1000, 2),
                    "duration_min": round(route["duration"] / 60, 1)
                }
    except Exception as e:
        print(f"[OSRM Router] Error fetching road route: {e}")

    return None
