import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="Geotechnická AI Analýza", layout="wide")
st.title("🧠 Geotechnický AI analyzátor hodnoty Kb")

st.markdown("Nahraj svoje CSV. Tento kód obsahuje agresivní čištění, které se s formátováním Excelu nemaže.")
uploaded_file = st.file_uploader("Vyber CSV soubor", type=['csv'])

if uploaded_file is not None:
    # 1. NAČTENÍ DAT (automatická detekce oddělovače)
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='windows-1250')
    
    st.write("### 1. Náhled surových dat (před čištěním)")
    st.dataframe(df.head(5), use_container_width=True)

    # --- BOČNÍ PANEL: PŘIŘAZENÍ SLOUPCŮ ---
    st.sidebar.header("1. Přiřazení sloupců")
    cols = df.columns.tolist()

    col_edef2 = st.sidebar.selectbox("Kde je Edef2?", cols, index=1 if len(cols)>1 else 0)
    col_evd = st.sidebar.selectbox("Kde je Evd?", cols, index=min(5, len(cols)-1))
    col_vlhkost = st.sidebar.selectbox("Kde je Vlhkost (w)?", cols, index=min(2, len(cols)-1))
    col_obj = st.sidebar.selectbox("Kde je Ro d?", cols, index=min(4, len(cols)-1))
    col_kb = st.sidebar.selectbox("Kde je Kb?", cols, index=min(6, len(cols)-1))

    # --- AGRESIVNÍ ČIŠTĚNÍ DAT (BULDOZER) ---
    df_model = df[[col_edef2, col_evd, col_vlhkost, col_obj, col_kb]].copy()
    
    def force_numeric(series):
        # Převede na text, smaže všechny mezery, nahradí čárky za tečky a vynutí číslo
        s = series.astype(str).str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce')

    for col in df_model.columns:
        df_model[col] = force_numeric(df_model[col])
    
    pocet_pred = len(df_model)
    df_model = df_model.dropna()
    pocet_po = len(df_model)

    # --- DEBUGGING (Když se to zase posere) ---
    if pocet_po < 20:
        st.error(f"A sakra! Zbylo jen {pocet_po} z {pocet_pred} řádků. Něco blokuje převod na čísla.")
        st.markdown("Podívej se do tabulky níže. Sloupce s `(Zkouška)` ukazují, co z toho Python udělal. Kde je **NaN**, tam narazil na znak, který nešlo převést na číslo (např. text v číselném sloupci).")
        
        df_debug = df[[col_edef2, col_evd, col_vlhkost, col_obj, col_kb]].copy()
        for col in df_debug.columns:
            df_debug[f"{col} (Zkouška)"] = force_numeric(df_debug[col])
            
        st.dataframe(df_debug, use_container_width=True)
        st.stop()
    else:
        st.success(f"Data vyčištěna! Připraveno {pocet_po} z {pocet_pred} řádků k analýze.")

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
                st.warning(f"V této skupině je jen {len(df_subset)} řádků. Zkus posunout slidery vlevo.")
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
                st.info("Algoritmus u této skupiny nenašel dost silnou závislost.")
else:
    st.info("Čekám na nahrání dat...")
