import json, math, time, requests, streamlit as st
import pandas as pd, numpy as np, ee, xml.etree.ElementTree as ET
from io import BytesIO
import py3dep, time
import random
import re

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="TAR",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================
# HEADER UI
# ================================================
st.markdown("""
<style>
.title { text-align: center; font-size: 30px; font-weight: bold; margin-top: -10px; }
.glitter {
    text-align: center; font-size: 20px; font-weight: bold;
    background: linear-gradient(90deg,#ff0000,#ff9900,#ffff00,#33cc33,#0066ff,#6600cc,#ff3399);
    background-size:400% 400%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation: glitter 5s ease infinite;
}
@keyframes glitter{0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
</style>
""", unsafe_allow_html=True)

logo_path = "logo.png"

col1, col2, col3 = st.columns([1,4,1])
with col1:
    st.image(logo_path, width=125)
with col2:
    st.markdown('<div class="title">Cube Highways Technologies Private Limited</div>', unsafe_allow_html=True)
    st.markdown('<div class="glitter">Technical Audit & Rating</div>', unsafe_allow_html=True)

# --- Authentication ---
def check_password():
    """Returns `True` if the user had a correct password."""
    if "password_correct" not in st.session_state:
        # First run, show input widgets
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        if st.button("Login"):
            if (
                st.session_state.username == st.secrets["auth"]["user"]
                and st.session_state.password == st.secrets["auth"]["password"]
            ):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("❌ Incorrect username or password")
        return False
    elif st.session_state.password_correct:
        return True
    return False

# Stop the app if incorrect
if not check_password():
    st.stop()

# ================================================
# AUTHENTICATE EARTH ENGINE
# ================================================
service_account = st.secrets["GEE"]["service_account"]
private_key = st.secrets["GEE"]["private_key"]

credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
ee.Initialize(credentials)
st.success("✅ Google Earth Engine authenticated successfully")

# ================================================
# MAPBOX TOKEN
# ================================================
MAPBOX_TOKEN = "sk.eyJ1IjoibmF2ZWVua3ViZW4iLCJhIjoiY21ncXZpZDBkMDlvbTJqczhxYTMyNXJ5ZCJ9.N3zxp8rqbp_M0ZtZeUgHfg"

# ================================================
# UPLOAD KML FILE
# ================================================
import zipfile
import io
import xml.etree.ElementTree as ET

uploaded_files = st.file_uploader(
    "📤 Upload alignment files (.kml or .kmz)",
    type=["kml", "kmz"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more KML or KMZ alignment files.")
    st.stop()

# Loop through each uploaded file
for file_index, uploaded in enumerate(uploaded_files, start=1):
    file_name = uploaded.name.replace(".kml", "").replace(".kmz", "")
    st.markdown(f"### 📄 ({file_index}) Processing file: `{file_name}`")

    try:
        # =========================
        # STEP 1: Read KML contents
        # =========================
        if uploaded.name.lower().endswith(".kmz"):
            with zipfile.ZipFile(io.BytesIO(uploaded.read()), 'r') as z:
                kml_files = [f for f in z.namelist() if f.endswith(".kml")]
                if not kml_files:
                    st.error(f"No .kml file found inside {uploaded.name}.")
                    continue
                kml_content = z.read(kml_files[0]).decode("utf-8")
        else:
            kml_content = uploaded.getvalue().decode("utf-8")

        # =========================
        # STEP 2: Parse KML features
        # =========================
        root = ET.fromstring(kml_content)
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        points = []

        for pm in root.findall(".//kml:Placemark", ns):
            name_elem = pm.find("kml:name", ns)
            coord_elem = pm.find(".//kml:coordinates", ns)

            # Skip if coordinates missing
            if coord_elem is None or not coord_elem.text.strip():
                continue

            coords = coord_elem.text.strip().split()
            geom_type = "LineString" if len(coords) > 1 else "Point"

            # Only extract Point coordinates
            if geom_type == "Point":
                lon, lat, *_ = [float(x) for x in coords[0].split(",")]
                name = name_elem.text.strip() if name_elem is not None else ""

                # Convert "0+100" to numeric for sorting
                name_numeric = 0.0
                try:
                    name_numeric = float(name.replace("+", "")) if "+" in name else float(name)
                except:
                    pass

                points.append({
                    "name": name,
                    "name_numeric": name_numeric,
                    "lat": lat,
                    "lon": lon
                })

        # =========================
        # STEP 3: Sort and confirm
        # =========================
        points_sorted = sorted(points, key=lambda p: p["name_numeric"])
        st.success(f"✅ Loaded {len(points_sorted)} chainage points from file: `{uploaded.name}`")

    except Exception as e:
        st.error(f"❌ KML/KMZ parsing failed for {uploaded.name}: {e}")
        continue
# ================================================
# FUNCTIONS
# ================================================
def get_elevation_slope(lat, lon):
    try:
        bounds = (lon - 0.0003, lat - 0.0003, lon + 0.0003, lat + 0.0003)
        elev_arr, res = py3dep.get_map("elevation", bounds, resolution=1, crs="EPSG:4326", return_array=True)
        elev = float(np.nanmean(elev_arr))
        gy, gx = np.gradient(elev_arr, res)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        return elev, float(np.nanmean(slope)), "USGS 3DEP"
    except Exception:
        try:
            dem = ee.Image("NASA/NASADEM_HGT/001")
            pt = ee.Geometry.Point([lon, lat])
            elev = dem.sample(pt, 30).first().get("elevation").getInfo()
            slope_img = ee.Terrain.slope(dem)
            slope = slope_img.sample(pt, 30).first().get("slope").getInfo()
            return elev, slope, "NASADEM"
        except Exception:
            return None, None, "None"

def classify_terrain(slope):
    if slope is None: return "Unknown"
    if slope < 6: return "Plain"
    elif slope < 15: return "Rolling"
    elif slope < 30: return "Hilly"
    return "Steep"

# --- Load datasets ---
worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S_10m").first().select("built_surface")
landcov = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").first().select(
    ["tree-coverfraction", "grass-coverfraction", "shrub-coverfraction"]
)

def is_near_road(lat, lon, radius=2):
    """
    Check if a point lies near any OSM road (within radius in meters).
    Used to avoid classifying highways as Urban in WorldCover.
    """
    q = f"""
    [out:json][timeout:10];
    way(around:{radius},{lat},{lon})[highway];
    out ids;
    """
    try:
        r = requests.get("https://overpass-api.de/api/interpreter", params={"data": q}, timeout=15)
        data = r.json().get("elements", [])
        return len(data) > 0
    except:
        return False


def classify_landuse(lat, lon):
    """
    Classify surroundings (Urban, Semi-Urban, Rural, Forest)
    by averaging conditions within a 100 m circle,
    and ignoring road surfaces falsely labeled as 'Built-up'.
    """
    try:
        pt = ee.Geometry.Point([lon, lat])
        buffer = pt.buffer(100) 

        # --- Compute built-up mean (GHSL 2023A) ---
        built_mean = ghsl.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e8
        ).get("built_surface").getInfo()
        built_mean = 0 if built_mean is None else float(built_mean)

        # --- Dominant WorldCover class (mode) ---
        wc_mode = worldcover.reduceRegion(
            reducer=ee.Reducer.mode(), geometry=buffer, scale=10, maxPixels=1e8
        ).get("Map").getInfo()
        wc_mode = int(wc_mode) if wc_mode is not None else 0

        # --- Mean vegetation fractions ---
        veg_stats = landcov.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=100,
            maxPixels=1e8
        )
        tree = veg_stats.get("tree-coverfraction").getInfo() or 0
        grass = veg_stats.get("grass-coverfraction").getInfo() or 0
        shrub = veg_stats.get("shrub-coverfraction").getInfo() or 0
        veg_total = tree + grass + shrub

        # --- Check proximity to road ---
        road_nearby = is_near_road(lat, lon)

        # --- Classification Logic ---
        if (built_mean > 10 and wc_mode == 50) and not road_nearby:
            category = "Urban"
        elif (7 <= built_mean <= 60 and wc_mode < 50):
            category = "Semi Urban"
        elif wc_mode in [10, 20] or veg_total >= 50:
            category = "Forest"
        else:
            category = "Rural"

        # --- Safety rule: downgrade Urban to Rural if only near road ---


        # --- Return all values ---
        return {
            "Category": category,
            "Built-up_%": round(built_mean, 2),
            "Dominant_Class": wc_mode,
            "Tree_%": round(tree, 2),
            "Grass_%": round(grass, 2),
            "Shrub_%": round(shrub, 2),
            
        }

    except Exception as e:
        return {"Category": "Unknown", "Error": str(e)}

def safe_ref(ref): 
    return str(ref).split(";")[0].strip().replace(" ", "") if ref else "-"


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter"
]

def query_road(lat, lon, retries=3):
    """
    Robust OSM Overpass query to get road tags consistently.
    Retries multiple Overpass mirrors with backoff if response is incomplete.
    """
    q = f"[out:json][timeout:60];way(around:100,{lat},{lon})[highway];out tags 1;"

    for attempt in range(retries):
        url = random.choice(OVERPASS_URLS)
        try:
            r = requests.get(url, params={"data": q}, timeout=60)
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue

            data = r.json().get("elements", [])
            if not data:
                time.sleep(2 * (attempt + 1))
                continue

            tags = data[0].get("tags", {})
            ref = safe_ref(tags.get("ref", "-"))
            if tags:
                return ref, tags

        except Exception as e:
            time.sleep(2 * (attempt + 1))  # exponential delay before retry

    # fallback if everything fails
    return "-", {}


def get_carriageway_type(tags, ref):
    if any(k in tags for k in ["lanes", "lanes:forward", "lanes:backward", "oneway", "divider", "dual_carriageway"]):
        return "Divided"
    if ref.startswith(("NH", "AH")):
        return "Divided"
    return "Undivided"


def query_crossings(lat, lon, radius=50, retries=3):
    q = f"""[out:json][timeout:60];
    (way(around:{radius},{lat},{lon})[bridge];
     node(around:{radius},{lat},{lon})[highway=crossing];);
    out tags 50;"""

    for attempt in range(retries):
        url = random.choice(OVERPASS_URLS)
        try:
            r = requests.get(url, params={"data": q}, timeout=60)
            if r.status_code == 200:
                data = r.json().get("elements", [])
                result = {"Bridge": False, "Crossing": False}
                for el in data:
                    t = el.get("tags", {})
                    if "bridge" in t: result["Bridge"] = True
                    if t.get("highway") == "crossing": result["Crossing"] = True
                if any(result.values()):
                    return "; ".join([k for k, v in result.items() if v])
        except:
            time.sleep(2 * (attempt + 1))
    return "-"


# ================================================
# SOIL DATA CONFIG
# ================================================
datasets = {
    "clay": {"path": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", "unit": "% (kg/kg)"},
    "sand": {"path": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", "unit": "% (kg/kg)"},
    "bulk_density": {"path": "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02", "unit": "x10 (kg/m³)"},
    "organic_carbon": {"path": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "unit": "x5 (g/kg)"},
    "ph": {"path": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02", "unit": "pHx10 in H2O"},
    "water_content": {"path": "OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01", "unit": "% (m³/m³)"},
}
depth_bands = ["b0", "b10", "b30", "b60", "b100", "b200"]

def get_soil_properties(lat, lon):
    pt = ee.Geometry.Point([lon, lat])
    result = {}
    for name, info in datasets.items():
        img = ee.Image(info["path"])
        for band in depth_bands:
            if band in img.bandNames().getInfo():
                try:
                    val = img.select(band).sample(pt, 250).first().get(band).getInfo()
                    result[f"{name}_{band} ({info['unit']})"] = val
                except:
                    result[f"{name}_{band} ({info['unit']})"] = None
    return result

# ================================================
# MAIN PROCESSING
# ================================================

zip_buffer = BytesIO()
excel_files = []
records = []
progress = st.progress(0)

for i, p in enumerate(points_sorted):
    lat, lon, name = p["lat"], p["lon"], p["name"]

    elev, slope, dem_src = get_elevation_slope(lat, lon)
    terrain = classify_terrain(slope)
    landuse = classify_landuse(lat, lon)
    landuse_category = landuse.get("Category", "Unknown")
    ref, tags = query_road(lat, lon)
    lanes = get_carriageway_type(tags, ref)
    crossings = query_crossings(lat, lon)
    soil_data = get_soil_properties(lat, lon)

    if terrain in ["Hilly", "Steep"]:
        remarks = "Challenging terrain, review alignment"
    elif landuse in ["Urban", "Semi Urban"]:
        remarks = "Feasible, minor shift may be needed"
    else:
        remarks = "Feasible"


    img_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},18,0/1080x720?access_token={MAPBOX_TOKEN}"

    rec = {
        "Chainage": name, "Latitude": lat, "Longitude": lon,
        "Road Number": ref, "Lane (Total)": f"{lanes} Lane",
        "Elevation (m)": elev, "Gradient %": slope, "Terrain Type": terrain,
        "Land Use" : landuse_category,
        "Land Use Detail": landuse, "Crossings / Structures nearby": crossings,
        "Remarks": remarks, "DEM Source": dem_src, "Google Earth View (~400 m)": img_url,
        "OSM Tags": json.dumps(tags, ensure_ascii=False)
    }
    rec.update(soil_data)
    records.append(rec)
    progress.progress((i + 1) / len(points_sorted))
    time.sleep(0.02)

st.success("✅ TAR")

# ================================================
# DISPLAY & DOWNLOAD
# ================================================
df = pd.DataFrame(records)
# --- Preview only first file ---
if file_index == 0:
    st.subheader(f"📄 Preview of {file_name}")
    st.dataframe(df.head(10), use_container_width=True)
# --- Save this Excel ---
buf = BytesIO()
df.to_excel(buf, index=False, engine="openpyxl")
excel_files.append((f"{file_name}.xlsx", buf))
# --- Progress update ---
progress_value = (file_index + 1) / len(uploaded_files)
progress_value = min(progress_value, 1.0)
progress.progress(progress_value)
st.success(f"✅ Completed: {file_name}")
# ================================================
# ZIP DOWNLOAD
# ================================================
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
    for filename, data in excel_files:
        data.seek(0)
        zipf.writestr(filename, data.read())

st.download_button(
    label="📦 Download All Excel Files (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="OFC_All_Outputs.zip",
    mime="application/zip"
)

