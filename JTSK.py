import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression

st.set_page_config(page_title="Geotechnická AI Analýza", layout="wide")
st.title("🧠 Geotechnický AI analyzátor hodnoty Kb")
st.markdown("Kód optimalizovaný pro český Excel formát (přesně dle tvého Sešit1.csv).")

uploaded_file = st.file_uploader("Vyber CSV soubor", type=['csv'])

if uploaded_file is not None:
    # 1. NAČTENÍ PŘESNĚ PRO TVŮJ FORMÁT (utf-8-sig, středník a desetinná čárka)
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='windows-1250')
    
    # Oprava prázdného názvu prvního sloupce s ID vzorků (STA01...)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Vzorek_ID'})

    # 2. PŘEVOD NA ČÍSLA (pro jistotu, kdyby se někde schoval text)
    for col in df.columns:
        if col != 'Vzorek_ID' and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write("### 1. Načtená data")
    st.dataframe(df.head(5), use_container_width=True)

    # --- BOČNÍ PANEL: PŘIŘAZENÍ SLOUPCŮ ---
    st.sidebar.header("1. Přiřazení sloupců")
    cols = df.columns.tolist()

    # Inteligentní předvýběr názvů z tvého souboru
    idx_edef2 = cols.index('Edef,2') if 'Edef,2' in cols else 0
    idx_evd = cols.index('Evd výběr') if 'Evd výběr' in cols else 0
    idx_w = cols.index('w') if 'w' in cols else 0
    idx_ro = cols.index('Ro d') if 'Ro d' in cols else 0
    idx_kb = cols.index('Kb') if 'Kb' in cols else 0

    col_edef2 = st.sidebar.selectbox("Kde je Edef2?", cols, index=idx_edef2)
    col_evd = st.sidebar.selectbox("Kde je Evd?", cols, index=idx_evd)
    col_vlhkost = st.sidebar.selectbox("Kde je Vlhkost (w)?", cols, index=idx_w)
    col_obj = st.sidebar.selectbox("Kde je Ro d?", cols, index=idx_ro)
    col_kb = st.sidebar.selectbox("Kde je Kb?", cols, index=idx_kb)

    df_model = df[[col_edef2, col_evd, col_vlhkost, col_obj, col_kb]].dropna().copy()
    st.success(f"Připraveno k analýze: {len(df_model)} z {len(df)} řádků.")

    if len(df_model) < 5:
        st.error("Máš extrémně málo čistých řádků. Analýza nemůže proběhnout.")
        st.stop()

    # --- BOČNÍ PANEL: TŘÍDĚNÍ ---
    st.sidebar.header("2. Nastavení třídění")
    tridici_sloupec = st.sidebar.selectbox("Podle čeho data rozdělit?", [col_vlhkost, col_obj])

    min_val = float(df_model[tridici_sloupec].min())
    max_val = float(df_model[tridici_sloupec].max())

    mezni_hodnota_1 = st.sidebar.slider("Hranice Nízká/Střední:", min_val, max_val, min_val + (max_val-min_val)*0.33)
    mezni_hodnota_2 = st.sidebar.slider("Hranice Střední/Vysoká:", min_val, max_val, min_val + (max_val-min_val)*0.66)

    hranice = [min_val - 1, mezni_hodnota_1, mezni_hodnota_2, max_val + 1]
    stitky = ['Nízká', 'Střední', 'Vysoká']
    df_model['Kategorie'] = pd.cut(df_model[tridici_sloupec], bins=hranice, labels=stitky)

    # --- VÝSLEDKY ---
    st.markdown("---")
    st.subheader("🔍 Výsledky analýzy (Korelace & Lasso Regrese)")

    tabs = st.tabs(stitky)

    for i, kategorie in enumerate(stitky):
        with tabs[i]:
            df_subset = df_model[df_model['Kategorie'] == kategorie].copy()
            
            if len(df_subset) < 4:
                st.warning(f"V této skupině je jen {len(df_subset)} řádků. K provedení analýzy posuň slidery vlevo.")
                continue
                
            st.write(f"Analyzuji na **{len(df_subset)} řádcích** dat...")
            
            X = df_subset[[col_edef2, col_evd, col_vlhkost, col_obj]].drop(columns=[tridici_sloupec], errors='ignore')
            y = df_subset[col_kb]
            
            # Vygenerování všech možných matematických interakcí (mocniny, křížení)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            vsechny_kombinace = poly.get_feature_names_out(X.columns)
            
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)
            
            # Lasso regrese (hledá nejsilnější vazby, zbytek srazí na 0)
            cv_folds = 3 if len(df_subset) >= 10 else 2
            lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=10000)
            lasso.fit(X_poly_scaled, y)
            
            vysledky = []
            for koef, nazev in zip(lasso.coef_, vsechny_kombinace):
                if abs(koef) > 0.0001: 
                    vysledky.append({"Vazba": nazev, "Vliv na Kb (Lasso)": koef})
            
            # Pokud Lasso kvůli malému vzorku zařízne úplně všechno, ukážeme aspoň standardní korelaci
            if not vysledky:
                st.info("Lasso regrese na takto malém vzorku nenašla absolutní prioritu. Zobrazuji proto běžné korelace s cílovou hodnotou:")
                
                # Obyčejná lineární regrese pro každý polynom zvlášť pro zjištění vlivu
                lin = LinearRegression()
                lin.fit(X_poly_scaled, y)
                
                for koef, nazev in zip(lin.coef_, vsechny_kombinace):
                    vysledky.append({"Vazba": nazev, "Vliv na Kb (Lasso)": koef})
                    
            # Vykreslení grafu výsledků
            df_vysledky = pd.DataFrame(vysledky)
            df_vysledky['Abs vliv'] = df_vysledky['Vliv na Kb (Lasso)'].abs()
            df_vysledky = df_vysledky.sort_values(by='Abs vliv', ascending=False).head(10).drop(columns=['Abs vliv'])
            
            fig = px.bar(
                df_vysledky, 
                x="Vliv na Kb (Lasso)", 
                y="Vazba", 
                orientation='h',
                color="Vliv na Kb (Lasso)",
                color_continuous_scale=px.colors.diverging.RdBu,
                title=f"TOP 10 závislostí pro Kb (Kategorie: {kategorie})"
            )
            # Otočení osy Y, aby nejsilnější vliv byl nahoře
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
else:
    st.info("Čekám na nahrání datového souboru...")
