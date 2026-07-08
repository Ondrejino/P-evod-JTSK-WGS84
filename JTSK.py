import streamlit as st
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="Bleskový Drtič Korelací", layout="wide")
st.title("⚡ Bleskový a inteligentní vyhledávač korelací")
st.markdown("Používá předvýběr (Feature Selection). Najde rovnice až o 4 neznámých i dynamické hranice za zlomek sekundy bez padání.")

uploaded_file = st.file_uploader("Nahraj tvůj soubor (Sešit1.csv)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='windows-1250')
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna()
    
    cols = df.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        idx_kb = cols.index('Kb') if 'Kb' in cols else len(cols)-1
        target = st.selectbox("Cílová hodnota (Y):", cols, index=idx_kb)
    with col2:
        default_features = [c for c in cols if c != target and c != 'w*w']
        features = st.multiselect("Základní vstupní hodnoty (X):", cols, default=default_features)

    if features:
        st.markdown("---")
        st.write("🔄 Analyzuji TOP kombinace...")
        
        # 1. GENERUJEME VŠECHNO
        df_math = df.copy()
        vsechny_promenne = list(features)
        
        for f in features:
            df_math[f"{f}^2"] = df_math[f] ** 2
            vsechny_promenne.append(f"{f}^2")
            
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                jmeno = f"{features[i]}*{features[j]}"
                df_math[jmeno] = df_math[features[i]] * df_math[features[j]]
                vsechny_promenne.append(jmeno)

        # 2. INTELIGENTNÍ PŘEDVÝBĚR (Tohle ušetří 25 minut)
        korelace = {}
        for prom in vsechny_promenne:
            if df_math[prom].std() > 0:
                korelace[prom] = df_math[prom].corr(df_math[target])
                
        # Vezmeme jen 8 nejsilnějších vlivů
        top_8_promennych = sorted(korelace, key=lambda x: abs(korelace[x]) if pd.notna(korelace[x]) else 0, reverse=True)[:8]
        
        vysledky = []
        
        def analyzuj_segment(df_vzorek, nazev_segmentu):
            if len(df_vzorek) < 6:
                return
                
            y = df_vzorek[target]
            max_promennych = min(4, max(1, len(df_vzorek) // 3))
            
            # Skládáme rovnice POUZE z elitních 8 proměnných
            for pocet_promennych in range(1, max_promennych + 1):
                for kombinace in itertools.combinations(top_8_promennych, pocet_promennych):
                    X_subset = df_vzorek[list(kombinace)]
                    
                    if X_subset.std().min() == 0:
                        continue 
                        
                    model = LinearRegression()
                    model.fit(X_subset, y)
                    y_pred = model.predict(X_subset)
                    r2_puvodni = r2_score(y, y_pred)
                    
                    if r2_puvodni < 0.4:
                        continue 
                    
                    casti = [f"{k:+.4f}*{n}" for k, n in zip(model.coef_, kombinace)]
                    rovnice = f"={model.intercept_:.4f} " + " ".join(casti)
                    
                    vysledky.append({
                        "Skupina": nazev_segmentu,
                        "R²": r2_puvodni,
                        "Vyhozeno extrémů": 0,
                        "Vzorků": len(df_vzorek),
                        "Parametry": " | ".join(kombinace),
                        "Rovnice": rovnice
                    })
                    
                    # OUTLIERS
                    rezidua = np.abs(y - y_pred)
                    odchylka = rezidua.std()
                    outlier_mask = rezidua > (1.5 * odchylka)
                    pocet_outlieru = outlier_mask.sum()
                    
                    if 0 < pocet_outlieru <= 2 and (len(df_vzorek) - pocet_outlieru) >= 5:
                        df_ciste = df_vzorek[~outlier_mask]
                        X_ciste = df_ciste[list(kombinace)]
                        y_ciste = df_ciste[target]
                        
                        model_cisty = LinearRegression()
                        model_cisty.fit(X_ciste, y_ciste)
                        r2_cisty = r2_score(y_ciste, model_cisty.predict(X_ciste))
                        
                        if r2_cisty > r2_puvodni + 0.05:
                            casti_ciste = [f"{k:+.4f}*{n}" for k, n in zip(model_cisty.coef_, kombinace)]
                            rovnice_ciste = f"={model_cisty.intercept_:.4f} " + " ".join(casti_ciste)
                            
                            vysledky.append({
                                "Skupina": f"{nazev_segmentu} (ČISTÉ)",
                                "R²": r2_cisty,
                                "Vyhozeno extrémů": pocet_outlieru,
                                "Vzorků": len(df_ciste),
                                "Parametry": " | ".join(kombinace),
                                "Rovnice": rovnice_ciste
                            })

        analyzuj_segment(df_math, "Všechna data")

        # Rozdělení dat
        for f in features:
            unikatni_hodnoty = sorted(df_math[f].unique())
            for hrana in unikatni_hodnoty[2:-2]:
                df_spodni = df_math[df_math[f] <= hrana]
                df_horni = df_math[df_math[f] > hrana]
                analyzuj_segment(df_spodni, f"Pokud {f} <= {hrana}")
                analyzuj_segment(df_horni, f"Pokud {f} > {hrana}")

        if vysledky:
            df_vysledky = pd.DataFrame(vysledky)
            # Ochrana před prázdným DataFramem a falešnými korelací
            df_vysledky = df_vysledky[df_vysledky["R²"] < 0.999]
            
            if not df_vysledky.empty:
                df_vysledky = df_vysledky.drop_duplicates(subset=["R²", "Parametry"])
                df_vysledky = df_vysledky.sort_values(by="R²", ascending=False).reset_index(drop=True)
                
                st.subheader("🏆 Výsledky")
                # Záměrně odebráno problematické .style.format - tabulka se vypíše 100% spolehlivě
                st.dataframe(df_vysledky, use_container_width=True, height=500)
                
                st.markdown("### 🔥 Absolutní vítěz:")
                nejlepsi = df_vysledky.iloc[0]
                st.success(f"Ideální hranici nalezl algoritmus zde: **{nejlepsi['Skupina']}**")
                st.info(f"R² = {nejlepsi['R²']:.4f} | Vzorků: {int(nejlepsi['Vzorků'])} | Smazáno extrémů: {int(nejlepsi['Vyhozeno extrémů'])}")
                st.code(nejlepsi['Rovnice'], language="excel")
            else:
                st.warning("Všechny nalezené modely byly buď příliš slabé, nebo šlo o falešné korelace malých vzorků.")
        else:
            st.warning("Algoritmus nenašel žádnou smysluplnou korelaci.")
