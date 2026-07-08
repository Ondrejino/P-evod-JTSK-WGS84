import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="AI Analýza Kb", layout="wide")
st.title("🧠 Geotechnický AI analyzátor hodnoty Kb")

# --- NAČTENÍ DAT (Zde si uprav pro své CSV) ---
@st.cache_data
def load_data():
    # Nasimulovaná data (nahraď pd.read_csv('tvoje_data.csv'))
    np.random.seed(42)
    return pd.DataFrame({
        'Edef2': np.random.uniform(20, 80, 200),
        'Evd': np.random.uniform(15, 60, 200),
        'vlhkost': np.random.uniform(3, 12, 200),
        'max_obj_hmotnost': np.random.uniform(1900, 2200, 200),
        'Kb': np.random.uniform(0.9, 1.05, 200)
    })

df = load_data()

# --- BOČNÍ PANEL (OVLÁDÁNÍ) ---
st.sidebar.header("⚙️ Nastavení logického třídění")

# Výběr, podle čeho chceme data dělit
tridici_sloupec = st.sidebar.selectbox(
    "Podle čeho chceš data rozdělit na segmenty?", 
    options=['vlhkost', 'max_obj_hmotnost']
)

# Dynamické posuvníky podle vybraného sloupce
min_val = float(df[tridici_sloupec].min())
max_val = float(df[tridici_sloupec].max())

st.sidebar.markdown(f"**Hranice pro {tridici_sloupec}**")
mezni_hodnota_1 = st.sidebar.slider("Hranice mezi Nízkou a Střední:", min_val, max_val, min_val + (max_val-min_val)*0.33)
mezni_hodnota_2 = st.sidebar.slider("Hranice mezi Střední a Vysokou:", min_val, max_val, min_val + (max_val-min_val)*0.66)

# Logika rozdělení
hranice = [min_val - 1, mezni_hodnota_1, mezni_hodnota_2, max_val + 1]
stitky = ['Nízká', 'Střední', 'Vysoká']
df['Kategorie'] = pd.cut(df[tridici_sloupec], bins=hranice, labels=stitky)

# --- HLAVNÍ PLOCHA ---
col1, col2, col3 = st.columns(3)
col1.metric("Celkem řádků", len(df))
col2.metric("Počet kategorií", len(stitky))
col3.metric("Cílová hodnota", "Kb")

st.markdown("---")
st.subheader("🔍 Výsledky Lasso regrese pro jednotlivé segmenty")

# Analyzujeme každou kategorii
tabs = st.tabs(stitky)

for i, kategorie in enumerate(stitky):
    with tabs[i]:
        df_subset = df[df['Kategorie'] == kategorie].copy()
        
        if len(df_subset) < 10:
            st.warning(f"V této kategorii je příliš málo dat ({len(df_subset)} řádků). Uprav posuvníky.")
            continue
            
        st.write(f"Analyzuji **{len(df_subset)} řádků** dat...")
        
        # Příprava proměnných (vše kromě Kb a Kategorizačních sloupců)
        X = df_subset[['Edef2', 'Evd', 'vlhkost', 'max_obj_hmotnost']].drop(columns=[tridici_sloupec])
        y = df_subset['Kb']
        
        # AI Srandičky (Mocniny a křížení)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        vsechny_kombinace = poly.get_feature_names_out(X.columns)
        
        # Škálování dat (nutnost pro Lasso)
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)
        
        # Lasso model hledá korelaci
        lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
        lasso.fit(X_poly_scaled, y)
        
        # Vytažení nejlepších závislostí
        vysledky = []
        for koeficient, nazev in zip(lasso.coef_, vsechny_kombinace):
            if abs(koeficient) > 0.0001:  # Vyřadíme to, co AI označila za zbytečné
                vysledky.append({"Proměnná (Kombinace)": nazev, "Vliv na Kb (Váha)": koeficient})
                
        if vysledky:
            df_vysledky = pd.DataFrame(vysledky)
            # Seřazení od největšího vlivu (absolutní hodnota)
            df_vysledky['Abs Vliv'] = df_vysledky['Vliv na Kb (Váha)'].abs()
            df_vysledky = df_vysledky.sort_values(by='Abs Vliv', ascending=False).drop(columns=['Abs Vliv'])
            
            # Cool vizualizace pomocí Plotly
            fig = px.bar(
                df_vysledky, 
                x="Vliv na Kb (Váha)", 
                y="Proměnná (Kombinace)", 
                orientation='h',
                title=f"Co nejvíce ovlivňuje Kb (Segment: {kategorie} {tridici_sloupec})",
                color="Vliv na Kb (Váha)",
                color_continuous_scale=px.colors.diverging.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_vysledky, use_container_width=True)
        else:
            st.info("Algoritmus nenašel žádnou silnou matematickou závislost. Zkus změnit rozdělení tříd.")
