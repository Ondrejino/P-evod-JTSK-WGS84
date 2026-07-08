import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="Geotechnická AI Analýza", layout="wide")
st.title("🧠 Geotechnický AI analyzátor hodnoty Kb")

# --- NAHRÁNÍ SOUBORU ---
st.markdown("Nahraj svoje CSV. Aplikace si sama poradí s tečkama, čárkama, středníkama i kódováním.")
uploaded_file = st.file_uploader("Vyber CSV soubor", type=['csv'])

if uploaded_file is not None:
    # 1. NEPRŮSTŘELNÉ NAČTENÍ DAT
    try:
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', encoding='windows-1250')
    
    # Pokud se to načetlo jen jako 1 sloupec, Excel použil jako oddělovač čárku
    if len(df.columns) == 1:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='windows-1250')

    # 2. FOOLPROOF ČIŠTĚNÍ DESETINNÝCH ČÁREK A TEČEK
    for col in df.columns:
        if df[col].dtype == 'object':
            # Nahradíme textové čárky za tečky a převedeme na čísla (texty jako STA01 ignoruje)
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='ignore')

    st.write("### Náhled tvých dat (opraveno)")
    st.dataframe(df.head(5), use_container_width=True)

    # --- BOČNÍ PANEL: PŘIŘAZENÍ SLOUPCŮ ---
    st.sidebar.header("1. Přiřazení sloupců")
    st.sidebar.markdown("Kde je co?")
    cols = df.columns.tolist()

    col_edef2 = st.sidebar.selectbox("Kde je Edef2?", cols, index=1 if len(cols)>1 else 0)
    col_evd = st.sidebar.selectbox("Kde je Evd?", cols, index=min(5, len(cols)-1))
    col_vlhkost = st.sidebar.selectbox("Kde je Vlhkost (w)?", cols, index=min(2, len(cols)-1))
    col_obj = st.sidebar.selectbox("Kde je Ro d?", cols, index=min(4, len(cols)-1))
    col_kb = st.sidebar.selectbox("Kde je Kb?", cols, index=min(6, len(cols)-1))

    # Připravíme modelová data a vyhodíme to, co se nepovedlo převést na čísla
    df_model = df[[col_edef2, col_evd, col_vlhkost, col_obj, col_kb]].copy()
    df_model = df_model.apply(pd.to_numeric, errors='coerce').dropna()

    if len(df_model) < 20:
        st.warning("Po vyčištění dat zbylo málo řádků. Zkontroluj přiřazení sloupců vlevo.")
        st.stop()

    # --- BOČNÍ PANEL: LOGICKÉ TŘÍDĚNÍ ---
    st.sidebar.header("2. Nastavení třídění")
    tridici_sloupec = st.sidebar.selectbox(
        "Podle čeho data rozdělit?", 
        options=[col_vlhkost, col_obj]
    )

    min_val = float(df_model[tridici_sloupec].min())
    max_val = float(df_model[tridici_sloupec].max())

    st.sidebar.markdown(f"**Nastav hranice pro: {tridici_sloupec}**")
    mezni_hodnota_1 = st.sidebar.slider("Hranice Nízká/Střední:", min_val, max_val, min_val + (max_val-min_val)*0.33)
    mezni_hodnota_2 = st.sidebar.slider("Hranice Střední/Vysoká:", min_val, max_val, min_val + (max_val-min_val)*0.66)

    hranice = [min_val - 1, mezni_hodnota_1, mezni_hodnota_2, max_val + 1]
    stitky = ['Nízká', 'Střední', 'Vysoká']
    df_model['Kategorie'] = pd.cut(df_model[tridici_sloupec], bins=hranice, labels=stitky)

    # --- HLAVNÍ PLOCHA: VÝSLEDKY ---
    st.markdown("---")
    st.subheader("🔍 Výsledky Lasso regrese")

    tabs = st.tabs(stitky)

    for i, kategorie in enumerate(stitky):
        with tabs[i]:
            df_subset = df_model[df_model['Kategorie'] == kategorie].copy()
            
            if len(df_subset) < 10:
                st.warning(f"Tady je jen {len(df_subset)} řádků. Zkus posunout slidery vlevo.")
                continue
                
            st.write(f"Analyzuji na **{len(df_subset)} řádcích** dat:")
            
            X = df_subset[[col_edef2, col_evd, col_vlhkost, col_obj]].drop(columns=[tridici_sloupec], errors='ignore')
            y = df_subset[col_kb]
            
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            vsechny_kombinace = poly.get_feature_names_out(X.columns)
            
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)
            
            lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
            lasso.fit(X_poly_scaled, y)
            
            vysledky = []
            for koeficient, nazev in zip(lasso.coef_, vsechny_kombinace):
                if abs(koeficient) > 0.0001: 
                    vysledky.append({"Vazba": nazev, "Vliv na Kb": koeficient})
                    
            if vysledky:
                df_vysledky = pd.DataFrame(vysledky)
                df_vysledky['Abs vliv'] = df_vysledky['Vliv na Kb'].abs()
                df_vysledky = df_vysledky.sort_values(by='Abs vliv', ascending=False).drop(columns=['Abs vliv'])
                
                fig = px.bar(
                    df_vysledky, 
                    x="Vliv na Kb", 
                    y="Vazba", 
                    orientation='h',
                    color="Vliv na Kb",
                    color_continuous_scale=px.colors.diverging.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Algoritmus u týhle skupiny nenašel dost silnou závislost.")
else:
    st.info("Čekám na nahrání dat...")
