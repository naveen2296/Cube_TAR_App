import json, math, time, requests, streamlit as st
import pandas as pd, numpy as np, ee, xml.etree.ElementTree as ET
from io import BytesIO
import py3dep, zipfile, io, random, re
from zipfile import ZipFile

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
    background-size:400% 400%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: glitter 5s ease infinite;
}
@keyframes glitter{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
</style>
""", unsafe_allow_html=True)

logo_path = "logo.png"
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image(logo_path, width=125)
with col2:
    st.markdown('<div class="title">Cube Highways Technologies Private Limited</div>', unsafe_allow_html=True)
    st.markdown('<div class="glitter">Technical Audit & Rating</div>', unsafe_allow_html=True)

# ================================================
# AUTHENTICATION
# ================================================
def check_password():
    if "password_correct" not in st.session_state:
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
    return st.session_state.password_correct

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

MAPBOX_TOKEN = "sk.eyJ1IjoibmF2ZWVua3ViZW4iLCJhIjoiY21ncXZpZDBkMDlvbTJqczhxYTMyNXJ5ZCJ9.N3zxp8rqbp_M0ZtZeUgHfg"

# ================================================
# MULTIPLE FILE UPLOAD
# ================================================
uploaded_files = st.file_uploader("📤 Upload multiple alignment files (.kml or .kmz)", type=["kml", "kmz"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload one or more KML or KMZ alignment files.")
    st.stop()

# ================================================
# HELPER FUNCTIONS
# ================================================
def parse_kml(uploaded):
    """Parse single KML/KMZ and return point list"""
    if uploaded.name.lower().endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(uploaded.read()), 'r') as z:
            kml_files = [f for f in z.namelist() if f.endswith(".kml")]
            if not kml_files:
                raise ValueError("No .kml file found inside KMZ.")
            kml_content = z.read(kml_files[0]).decode("utf-8")
    else:
        kml_content = uploaded.getvalue().decode("utf-8")

    root = ET.fromstring(kml_content)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    points = []
    for pm in root.findall(".//kml:Placemark", ns):
        name_elem = pm.find("kml:name", ns)
        coord_elem = pm.find(".//kml:coordinates", ns)
        if coord_elem is None or not coord_elem.text.strip():
            continue
        coords = coord_elem.text.strip().split()
        if len(coords) == 1:
            lon, lat, *_ = [float(x) for x in coords[0].split(",")]
            name = name_elem.text.strip() if name_elem is not None else ""
            try:
                name_numeric = float(name.replace("+", "")) if "+" in name else float(name)
            except:
                name_numeric = 0.0
            points.append({
                "name": name,
                "name_numeric": name_numeric,
                "lat": lat,
                "lon": lon
            })
    return sorted(points, key=lambda p: p["name_numeric"])

# ================================================
# SOIL & TERRAIN FUNCTIONS
# ================================================
def get_elevation_slope(lat, lon):
    try:
        bounds = (lon - 0.0003, lat - 0.0003, lon + 0.0003, lat + 0.0003)
        elev_arr, res = py3dep.get_map("elevation", bounds, resolution=1, crs="EPSG:4326", return_array=True)
        elev = float(np.nanmean(elev_arr))
        gy, gx = np.gradient(elev_arr, res)
        slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        return elev, float(np.nanmean(slope)), "USGS 3DEP"
    except:
        try:
            dem = ee.Image("NASA/NASADEM_HGT/001")
            pt = ee.Geometry.Point([lon, lat])
            elev = dem.sample(pt, 30).first().get("elevation").getInfo()
            slope_img = ee.Terrain.slope(dem)
            slope = slope_img.sample(pt, 30).first().get("slope").getInfo()
            return elev, slope, "NASADEM"
        except:
            return None, None, "None"

def classify_terrain(slope):
    if slope is None: return "Unknown"
    if slope < 6: return "Plain"
    elif slope < 15: return "Rolling"
    elif slope < 30: return "Hilly"
    return "Steep"

worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S_10m").first().select("built_surface")
landcov = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").first().select(
    ["tree-coverfraction", "grass-coverfraction", "shrub-coverfraction"]
)

def classify_landuse(lat, lon):
    try:
        pt = ee.Geometry.Point([lon, lat])
        buffer = pt.buffer(100)
        built_mean = ghsl.reduceRegion(ee.Reducer.mean(), geometry=buffer, scale=10).get("built_surface").getInfo() or 0
        wc_mode = worldcover.reduceRegion(ee.Reducer.mode(), geometry=buffer, scale=10).get("Map").getInfo() or 0
        veg_stats = landcov.reduceRegion(ee.Reducer.mean(), geometry=buffer, scale=100)
        tree = veg_stats.get("tree-coverfraction").getInfo() or 0
        grass = veg_stats.get("grass-coverfraction").getInfo() or 0
        shrub = veg_stats.get("shrub-coverfraction").getInfo() or 0
        veg_total = tree + grass + shrub
        if (built_mean > 10 and wc_mode == 50):
            category = "Urban"
        elif (7 <= built_mean <= 60 and wc_mode < 50):
            category = "Semi Urban"
        elif wc_mode in [10, 20] or veg_total >= 50:
            category = "Forest"
        else:
            category = "Rural"
        return {"Category": category, "Built-up_%": built_mean, "Tree_%": tree, "Grass_%": grass, "Shrub_%": shrub}
    except Exception as e:
        return {"Category": "Unknown", "Error": str(e)}

datasets = {
    "clay": {"path": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", "unit": "%"},
    "sand": {"path": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", "unit": "%"},
    "bulk_density": {"path": "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02", "unit": "kg/m³"},
    "organic_carbon": {"path": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "unit": "g/kg"},
    "ph": {"path": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02", "unit": "pH"},
    "water_content": {"path": "OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01", "unit": "%"},
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
# PROCESS FILES AND EXPORT ZIP
# ================================================
zip_buffer = BytesIO()
with ZipFile(zip_buffer, "w") as zipf:
    for idx, uploaded in enumerate(uploaded_files):
        st.markdown(f"### ⏳ Processing file: `{uploaded.name}` ...")
        try:
            points_sorted = parse_kml(uploaded)
            records = []
            progress = st.progress(0)

            for i, p in enumerate(points_sorted):
                lat, lon, name = p["lat"], p["lon"], p["name"]
                elev, slope, dem_src = get_elevation_slope(lat, lon)
                terrain = classify_terrain(slope)
                landuse = classify_landuse(lat, lon)
                soil_data = get_soil_properties(lat, lon)
                remarks = ("Challenging terrain, review alignment" if terrain in ["Hilly", "Steep"]
                           else "Feasible, minor shift may be needed" if landuse.get("Category") in ["Urban", "Semi Urban"]
                           else "Feasible")


                rec = {
                    "Chainage": name,
                    "Latitude": lat,
                    "Longitude": lon,
                    "Elevation (m)": elev,
                    "Gradient %": slope,
                    "Terrain Type": terrain,
                    "Land Use": landuse.get("Category", "Unknown"),
                    "Remarks": remarks,
                    "DEM Source": dem_src
                }
                rec.update(landuse)
                rec.update(soil_data)
                records.append(rec)
                progress.progress((i + 1) / len(points_sorted))

            df = pd.DataFrame(records)
            excel_buf = BytesIO()
            df.to_excel(excel_buf, index=False, engine="openpyxl")
            zipf.writestr(f"{uploaded.name.split('.')[0]}.xlsx", excel_buf.getvalue())

            if idx == 0:  # preview only first file
                st.dataframe(df.head(10), use_container_width=True)

            st.success(f"✅ Completed: {uploaded.name} ({len(df)} points)")

        except Exception as e:
            st.error(f"❌ Failed to process {uploaded.name}: {e}")

# ================================================
# ZIP DOWNLOAD
# ================================================
zip_buffer.seek(0)
st.download_button(
    "📦 Download All Excel Files (ZIP)",
    zip_buffer,
    file_name="TAR_All_Results.zip",
    mime="application/zip"
)
