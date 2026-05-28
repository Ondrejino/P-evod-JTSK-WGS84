import streamlit as st
from pyproj import Transformer

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="Geodetický Převodník 3D", layout="centered")
st.title("🌍 Ultimátní 3D Převodník: S-JTSK ↔ WGS84")

# --- POMOCNÉ FUNKCE ---
def decimal_to_dms(decimal_degree, is_lat):
    """Převede desetinné stupně (DD) na Stupně, minuty, vteřiny (DMS)."""
    if is_lat:
        direction = 'N' if decimal_degree >= 0 else 'S'
    else:
        direction = 'E' if decimal_degree >= 0 else 'W'
        
    abs_dd = abs(decimal_degree)
    degrees = int(abs_dd)
    minutes = int((abs_dd - degrees) * 60)
    seconds = (abs_dd - degrees - minutes/60) * 3600
    return f"{degrees}° {minutes}' {seconds:.5f}\" {direction}"

def dms_to_decimal(degrees, minutes, seconds, direction):
    """Převede Stupně, minuty, vteřiny (DMS) na desetinné stupně (DD)."""
    decimal = float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

# --- INICIALIZACE TRANSFORMÁTORŮ (Oprava 3D výšek) ---
@st.cache_resource
def get_transformers():
    # EPSG:5514+5705 = S-JTSK + Balt po vyrovnání (3D)
    # EPSG:4979 = WGS84 3D
    t_to_wgs = Transformer.from_crs("EPSG:5514+5705", "EPSG:4979", always_xy=True)
    t_to_jtsk = Transformer.from_crs("EPSG:4979", "EPSG:5514+5705", always_xy=True)
    return t_to_wgs, t_to_jtsk

transformer_to_wgs, transformer_to_jtsk = get_transformers()

# --- UŽIVATELSKÉ ROZHRANÍ ---
tab1, tab2, tab3 = st.tabs(["➡️ S-JTSK na WGS84", "⬅️ WGS84 (Desetinné) na S-JTSK", "⬅️ WGS84 (Stupně/Minuty) na S-JTSK"])

# ZÁLOŽKA 1: S-JTSK -> WGS84
with tab1:
    st.subheader("Převod z Křováka (S-JTSK) do GPS (WGS84)")
    st.info("Osa Y a X se v S-JTSK zadává se ZÁPORNÝM znaménkem. Výška Z je v metrech n. m. (Bpv).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        jtsk_y = st.number_input("Souřadnice Y (S-JTSK)", value=-600000.00, step=10.0, format="%.2f")
    with col2:
        jtsk_x = st.number_input("Souřadnice X (S-JTSK)", value=-1050000.00, step=10.0, format="%.2f")
    with col3:
        jtsk_z = st.number_input("Výška Z (Bpv v metrech)", value=249.00, step=1.0, format="%.2f")
        
    if st.button("Převést S-JTSK ➡️ WGS84", type="primary"):
        wgs_lon, wgs_lat, wgs_alt = transformer_to_wgs.transform(jtsk_y, jtsk_x, jtsk_z)
        st.success("✅ Převod dokončen!")
        st.markdown(f"""
        **Výstup WGS84 (Desetinné stupně):**
        * **Lat (Šířka):** `{wgs_lat:.7f}`
        * **Lon (Délka):** `{wgs_lon:.7f}`
        * **Elipsoidická výška:** `{wgs_alt:.3f} m`
        
        **Výstup WGS84 (Stupně/Minuty/Vteřiny):**
        * **Lat:** `{decimal_to_dms(wgs_lat, is_lat=True)}`
        * **Lon:** `{decimal_to_dms(wgs_lon, is_lat=False)}`
        """)

# ZÁLOŽKA 2: WGS84 (Desetinné) -> S-JTSK
with tab2:
    st.subheader("WGS84 (Desetinné formáty) do S-JTSK")
    col4, col5, col6 = st.columns(3)
    with col4:
        wgs_lat_dec = st.number_input("Zeměpisná šířka (Lat)", value=49.8248881, step=0.0001, format="%.7f")
    with col5:
        wgs_lon_dec = st.number_input("Zeměpisná délka (Lon)", value=15.3025219, step=0.0001, format="%.7f")
    with col6:
        wgs_alt_dec = st.number_input("GPS Výška (m)", value=293.30, step=1.0, format="%.2f")
        
    if st.button("Převést WGS84 (Desetinné) ➡️ S-JTSK", type="primary"):
        y_out, x_out, z_out = transformer_to_jtsk.transform(wgs_lon_dec, wgs_lat_dec, wgs_alt_dec)
        st.success("✅ Převod dokončen!")
        st.markdown(f"**S-JTSK:** Y: `{y_out:.3f}` | X: `{x_out:.3f}` | Výška Bpv: `{z_out:.3f} m`")

# ZÁLOŽKA 3: WGS84 (DMS) -> S-JTSK
with tab3:
    st.subheader("WGS84 (Stupně, Minuty, Vteřiny) do S-JTSK")
    st.markdown("Zeměpisná šířka (N/S)")
    col_lat1, col_lat2, col_lat3, col_lat4 = st.columns([1, 1, 1, 1])
    with col_lat1: lat_deg = st.number_input("Stupně (°)", value=49, step=1, key='lat_d')
    with col_lat2: lat_min = st.number_input("Minuty (')", value=49, step=1, key='lat_m')
    with col_lat3: lat_sec = st.number_input("Vteřiny (\")", value=29.59716, step=0.01, key='lat_s', format="%.5f")
    with col_lat4: lat_dir = st.selectbox("Směr", ["N", "S"], key='lat_dir')

    st.markdown("Zeměpisná délka (E/W)")
    col_lon1, col_lon2, col_lon3, col_lon4 = st.columns([1, 1, 1, 1])
    with col_lon1: lon_deg = st.number_input("Stupně (°)", value=15, step=1, key='lon_d')
    with col_lon2: lon_min = st.number_input("Minuty (')", value=18, step=1, key='lon_m')
    with col_lon3: lon_sec = st.number_input("Vteřiny (\")", value=9.07884, step=0.01, key='lon_s', format="%.5f")
    with col_lon4: lon_dir = st.selectbox("Směr", ["E", "W"], key='lon_dir')
    
    st.markdown("Výška")
    wgs_alt_dms = st.number_input("GPS Výška (m)", value=293.30, step=1.0, format="%.2f", key='alt_dms')

    if st.button("Převést WGS84 (DMS) ➡️ S-JTSK", type="primary"):
        lat_dec = dms_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
        lon_dec = dms_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)
        
        y_out, x_out, z_out = transformer_to_jtsk.transform(lon_dec, lat_dec, wgs_alt_dms)
        st.success("✅ Převod dokončen!")
        st.markdown(f"**Vypočtené desetinné stupně:** Lat: `{lat_dec:.7f}`, Lon: `{lon_dec:.7f}`")
        st.markdown(f"**S-JTSK:** Y: `{y_out:.3f}` | X: `{x_out:.3f}` | Výška Bpv: `{z_out:.3f} m`")
