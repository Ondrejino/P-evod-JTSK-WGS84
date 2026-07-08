import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="Geotechnická AI Analýza", layout="wide")
st.title("🧠 Geotechnický AI analyzátor hodnoty Kb")

# --- NAHRÁNÍ SOUBORU ---
st.markdown("Nahrajte svůj CSV soubor. Aplikace si sama poradí s daty a umožní vám vybrat sloupce pro analýzu.")
uploaded_file = st.file_uploader("Vyberte CSV soubor (vyexportovaný z Excelu)", type=['csv'])

if uploaded_file is not None:
    try:
        # engine='python' a sep=None umožní pandas automaticky poznat, zda je oddělovač čárka nebo středník
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Nepodařilo se načíst soubor. Zkontrolujte, zda je to platné CSV. Detail chyby: {e}")
        st.stop()

    st.write("### Náhled vašich dat")
    st.dataframe(df.head(5), use_container_width=True)

    # --- BOČNÍ PANEL: PŘIŘAZENÍ SLOUPCŮ ---
    st.sidebar.header("1. Přiřazení sloupců")
    st.sidebar.markdown("Řekněte algoritmu, kde najde které hodnoty:")
    cols = df.columns.tolist()

    # Uživatelské přiřazení sloupců k proměnným
    col_edef2 = st.sidebar.selectbox("Kde je Edef2?", cols, index=0)
    col_evd = st.sidebar.selectbox("Kde je Evd?", cols, index=min(1, len(cols)-1))
    col_vlhkost = st.sidebar.selectbox("Kde je Vlhkost?", cols, index=min(2, len(cols)-1))
    col_obj = st.sidebar.selectbox("Kde je Max. objemová hmotnost?", cols, index=min(3, len(cols)-1))
    col_kb = st.sidebar.selectbox("Kde je cílové Kb?", cols, index=min(4, len(cols)-1))

    # --- PŘÍPRAVA DAT PRO MODEL ---
    # Vybereme jen potřebné sloupce, převedeme na čísla a zahodíme řádky, kde něco chybí
    df_model = df[[col_edef2, col_evd, col_vlhkost, col_obj, col_kb]].copy()
    df_model = df_model.apply(pd.to_numeric, errors='coerce').dropna()

    if len(df_model) < 20:
        st.warning("Po vyčištění dat zbylo příliš málo řádků pro smysluplnou AI analýzu.")
        st.stop()

    # --- BOČNÍ PANEL: LOGICKÉ TŘÍDĚNÍ ---
    st.sidebar.header("2. Nastavení třídění")
    tridici_sloupec = st.sidebar.selectbox(
        "Podle čeho chceš data rozdělit na segmenty?", 
        options=[col_vlhkost, col_obj]
    )

    min_val = float(df_model[tridici_sloupec].min())
    max_val = float(df_model[tridici_sloupec].max())

    st.sidebar.markdown(f"**Nastavte hranice pro: {tridici_sloupec}**")
    mezni_hodnota_1 = st.sidebar.slider("Hranice mezi Nízkou a Střední hodnotou:", min_val, max_val, min_val + (max_val-min_val)*0.33)
    mezni_hodnota_2 = st.sidebar.slider("Hranice mezi Střední a Vysokou hodnotou:", min_val, max_val, min_val + (max_val-min_val)*0.66)

    # Rozdělení do kategorií
    hranice = [min_val - 1, mezni_hodnota_1, mezni_hodnota_2, max_val + 1]
    stitky = ['Nízká', 'Střední', 'Vysoká']
    df_model['Kategorie'] = pd.cut(df_model[tridici_sloupec], bins=hranice, labels=stitky)

    # --- HLAVNÍ PLOCHA: VÝSLEDKY ---
    st.markdown("---")
    st.subheader("🔍 Výsledky Lasso regrese pro jednotlivé segmenty")

    tabs = st.tabs(stitky)

    for i, kategorie in enumerate(stitky):
        with tabs[i]:
            df_subset = df_model[df_model['Kategorie'] == kategorie].copy()
            
            if len(df_subset) < 10:
                st.warning(f"V této kategorii je pouze {len(df_subset)} řádků. Upravte posuvníky vlevo pro lepší rozložení.")
                continue
                
            st.write(f"Závislosti nalezené na **{len(df_subset)} řádcích** dat:")
            
            # X jsou vstupy, y je výstup (Kb)
            X = df_subset[[col_edef2, col_evd, col_vlhkost, col_obj]].drop(columns=[tridici_sloupec], errors='ignore')
            y = df_subset[col_kb]
            
            # Automatické křížení a mocniny
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            vsechny_kombinace = poly.get_feature_names_out(X.columns)
            
            # Škálování a Lasso regrese
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)
            
            lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
            lasso.fit(X_poly_scaled, y)
            
            # Filtrace výsledků
            vysledky = []
            for koeficient, nazev in zip(lasso.coef_, vsechny_kombinace):
                if abs(koeficient) > 0.0001: 
                    vysledky.append({"Matematická vazba": nazev, "Síla vlivu na Kb": koeficient})
                    
            if vysledky:
                df_vysledky = pd.DataFrame(vysledky)
                df_vysledky['Absolutní síla'] = df_vysledky['Síla vlivu na Kb'].abs()
                df_vysledky = df_vysledky.sort_values(by='Absolutní síla', ascending=False).drop(columns=['Absolutní síla'])
                
                # Vykreslení grafu
                fig = px.bar(
                    df_vysledky, 
                    x="Síla vlivu na Kb", 
                    y="Matematická vazba", 
                    orientation='h',
                    color="Síla vlivu na Kb",
                    color_continuous_scale=px.colors.diverging.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pro tuto skupinu nenašel algoritmus žádnou dostatečně silnou matematickou závislost.")
else:
    st.info("Čekám na nahrání CSV souboru...")
