import streamlit as st
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Hledání nejlepší rovnice", layout="wide")
st.title("🤖 Automatické hledání vícenásobné regrese")
st.markdown("Tento kód zkouší **všechny možné kombinace více sloupců najednou** (od 1 až po všechny) a hledá tu s nejvyšším R².")

uploaded_file = st.file_uploader("Nahraj tvůj soubor (Sešit1.csv)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='windows-1250')
    
    # Čištění
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna()
    st.success(f"Načteno a vyčištěno: {len(df)} řádků.")

    # Výběr sloupců
    cols = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        idx_kb = cols.index('Kb') if 'Kb' in cols else len(cols)-1
        target = st.selectbox("Co počítáme (Cílová hodnota Y):", cols, index=idx_kb)
        
    with col2:
        default_features = [c for c in cols if c != target and c != 'w*w']
        features = st.multiselect("Z čeho můžeme počítat (Vstupní X):", cols, default=default_features)

    if features:
        st.markdown("---")
        st.subheader(f"🏆 Žebříček nejlepších rovnic pro výpočet {target}")
        
        y = df[target]
        vysledky = []
        
        # Jádro pudla: Itertools vyzkouší kombinace 1 sloupce, pak 2, pak 3, atd.
        for pocet_promennych in range(1, len(features) + 1):
            for kombinace in itertools.combinations(features, pocet_promennych):
                X_subset = df[list(kombinace)]
                
                # Výpočet regrese pro danou skupinu sloupců
                model = LinearRegression()
                model.fit(X_subset, y)
                y_pred = model.predict(X_subset)
                
                r2 = r2_score(y, y_pred)
                
                # Sestavení textu rovnice ve formátu pro Excel
                casti_rovnice = []
                for koeficient, nazev_sloupce in zip(model.coef_, kombinace):
                    casti_rovnice.append(f"{koeficient:+.4f}*{nazev_sloupce}")
                
                text_rovnice = f"={model.intercept_:.4f} " + " ".join(casti_rovnice)
                
                vysledky.append({
                    "Počet parametrů": pocet_promennych,
                    "Zahrnuté sloupce": ", ".join(kombinace),
                    "Spolehlivost (R²)": r2,
                    "Přesná rovnice": text_rovnice
                })
                
        # Zobrazení výsledků v tabulce, seřazeno od nejlepšího R²
        df_vysledky = pd.DataFrame(vysledky)
        df_vysledky = df_vysledky.sort_values(by="Spolehlivost (R²)", ascending=False).reset_index(drop=True)
        
        # Nastavení hezkého formátování tabulky ve Streamlitu
        st.dataframe(
            df_vysledky.style.format({"Spolehlivost (R²)": "{:.4f}"}),
            use_container_width=True,
            height=400
        )
        
        # Zobrazení úplně té nejlepší rovnice hezky velkým písmem
        st.markdown("### 🔥 Absolutní vítěz:")
        nejlepsi = df_vysledky.iloc[0]
        st.info(f"**R² = {nejlepsi['Spolehlivost (R²)']:.4f}**")
        st.code(nejlepsi['Přesná rovnice'], language="excel")
