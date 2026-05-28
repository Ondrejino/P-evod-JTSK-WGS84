import streamlit as st
from pyproj import Transformer

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="Geodetický Převodník", layout="centered")
st.title("🌍 3D Převodník: S-JTSK ↔ WGS84")

# --- POMOCNÉ FUNKCE PRO FORMÁTOVÁNÍ ---
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

# --- INICIALIZACE TRANSFORMÁTORŮ ---
# always_xy=True zajišťuje pořadí [X/Délka, Y/Šířka, Z/Výška]
@st.cache_resource
def get_transformers():
    t_to_wgs = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
    t_to_jtsk = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
    return t_to_wgs, t_to_jtsk

transformer_to_wgs, transformer_to_jtsk = get_transformers()

# --- UŽIVATELSKÉ ROZHRANÍ ---
tab1, tab2 = st.tabs(["➡️ S-JTSK na WGS84", "⬅️ WGS84 na S-JTSK"])

# ZÁLOŽKA 1: S-JTSK -> WGS84
with tab1:
    st.subheader("Převod z Křováka (S-JTSK) do GPS (WGS84)")
    st.info("Nezapomeň zadávat souřadnice X a Y se ZÁPORNÝM znaménkem (např. -600000).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        jtsk_y = st.number_input("Souřadnice Y (S-JTSK)", value=-600000.00, step=100.0, format="%.2f")
    with col2:
        jtsk_x = st.number_input("Souřadnice X (S-JTSK)", value=-1050000.00, step=100.0, format="%.2f")
    with col3:
        jtsk_z = st.number_input("Výška Z (m n. m.)", value=249.00, step=1.0, format="%.2f")
        
    if st.button("Převést na WGS84", type="primary"):
        wgs_lon, wgs_lat, wgs_alt = transformer_to_wgs.transform(jtsk_y, jtsk_x, jtsk_z)
        
        st.success("✅ Převod dokončen!")
        st.markdown(f"""
        **Desetinné stupně:**
        * **Zeměpisná šířka (Lat):** `{wgs_lat:.7f}`
        * **Zeměpisná délka (Lon):** `{wgs_lon:.7f}`
        * **Elipsoidická výška:** `{wgs_alt:.3f} m`
        
        **Stupně, minuty, vteřiny (DMS):**
        * **Lat:** `{decimal_to_dms(wgs_lat, is_lat=True)}`
        * **Lon:** `{decimal_to_dms(wgs_lon, is_lat=False)}`
        """)

# ZÁLOŽKA 2: WGS84 -> S-JTSK
with tab2:
    st.subheader("Převod z GPS (WGS84) do Křováka (S-JTSK)")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        wgs_lat_in = st.number_input("Zeměpisná šířka (Lat)", value=49.8248881, step=0.0001, format="%.7f")
    with col5:
        wgs_lon_in = st.number_input("Zeměpisná délka (Lon)", value=15.3025219, step=0.0001, format="%.7f")
    with col6:
        wgs_alt_in = st.number_input("GPS Výška (m)", value=293.30, step=1.0, format="%.2f")
        
    if st.button("Převést na S-JTSK", type="primary"):
        jtsk_y_out, jtsk_x_out, jtsk_z_out = transformer_to_jtsk.transform(wgs_lon_in, wgs_lat_in, wgs_alt_in)
        
        st.success("✅ Převod dokončen!")
        st.markdown(f"""
        **S-JTSK Souřadnice:**
        * **Y:** `{jtsk_y_out:.3f}`
        * **X:** `{jtsk_x_out:.3f}`
        * **Výška (Z):** `{jtsk_z_out:.3f} m`
        """)
