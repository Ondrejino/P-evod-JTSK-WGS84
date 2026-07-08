import streamlit as st
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="AI Drtič Korelací", layout="wide")
st.title("🤖 Ultimátní vyhledávač skrytých korelací")
st.markdown("Automaticky generuje matematické kombinace (mocniny, násobky) a hledá tu absolutně nejlepší přirozenou hranici pro rozdělení dat.")

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
        st.write(f"🔄 Generuji matematické operace a chroustám data...")
        
        # 1. VYTVOŘENÍ MATEMATICKÝCH KOMBINACÍ (Mocniny a násobení)
        poly_features = list(features)
        df_math = df.copy()
        
        # Mocniny (na druhou)
        for f in features:
            jmeno = f"{f}^2"
            df_math[jmeno] = df_math[f] ** 2
            poly_features.append(jmeno)
            
        # Křížení (násobení každý s každým)
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                jmeno = f"{features[i]}*{features[j]}"
                df_math[jmeno] = df_math[features[i]] * df_math[features[j]]
                poly_features.append(jmeno)

        vysledky = []
        
        # Jádro: Testování kombinací rovnic na jakémkoliv kousku dat
        def analyzuj_segment(df_vzorek, nazev_segmentu):
            # Ochrana: Potřebujeme aspoň 6 řádků, aby to byla statistika a ne věštění
            if len(df_vzorek) < 6:
                return
                
            y = df_vzorek[target]
            
            # Aby rovnice nebyla delší než samotná data (ochrana proti falešnému R^2 = 1.0)
            max_promennych = min(4, max(1, len(df_vzorek) // 3))
            
            for pocet_promennych in range(1, max_promennych + 1):
                for kombinace in itertools.combinations(poly_features, pocet_promennych):
                    X_subset = df_vzorek[list(kombinace)]
                    
                    if X_subset.std().min() == 0:
                        continue # Přeskočíme, pokud jsou v datech samé nuly
                        
                    # Standardní regrese
                    model = LinearRegression()
                    model.fit(X_subset, y)
                    y_pred = model.predict(X_subset)
                    r2_puvodni = r2_score(y, y_pred)
                    
                    if r2_puvodni < 0.3:
                        continue # Ignorujeme naprostý odpad, ať nezahltíme paměť
                    
                    casti = [f"{k:+.4f}*{n}" for k, n in zip(model.coef_, kombinace)]
                    rovnice = f"={model.intercept_:.4f} " + " ".join(casti)
                    
                    vysledky.append({
                        "Skupina": nazev_segmentu,
                        "R²": r2_puvodni,
                        "Vyhozeno extrémů": 0,
                        "Vzorků": len(df_vzorek),
                        "Parametry": " | ".join(kombinace),
                        "Rovnice pro Excel": rovnice
                    })
                    
                    # OUTLIERS: Detekce ustřelených bodů
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
                        y_pred_cisty = model_cisty.predict(X_ciste)
                        r2_cisty = r2_score(y_ciste, y_pred_cisty)
                        
                        # Zapsat jen když se to opravdu dramaticky zlepší
                        if r2_cisty > r2_puvodni + 0.05:
                            casti_ciste = [f"{k:+.4f}*{n}" for k, n in zip(model_cisty.coef_, kombinace)]
                            rovnice_ciste = f"={model_cisty.intercept_:.4f} " + " ".join(casti_ciste)
                            
                            vysledky.append({
                                "Skupina": f"{nazev_segmentu} (ČISTÉ)",
                                "R²": r2_cisty,
                                "Vyhozeno extrémů": pocet_outlieru,
                                "Vzorků": len(df_ciste),
                                "Parametry": " | ".join(kombinace),
                                "Rovnice pro Excel": rovnice_ciste
                            })

        # --- 1. ZKOUMÁME CELÝ DATASET ---
        analyzuj_segment(df_math, "Všechna data")

        # --- 2. INTELIGENTNÍ HLEDÁNÍ HRANICE ---
        # Zkoušíme rozříznout dataset v každé existující hodnotě
        for f in features:
            unikatni_hodnoty = sorted(df_math[f].unique())
            
            # Přeskočíme extrémní kraje, kde by zbyly třeba jen 2 body
            for hrana in unikatni_hodnoty[3:-3]:
                df_spodni = df_math[df_math[f] <= hrana]
                df_horni = df_math[df_math[f] > hrana]
                
                analyzuj_segment(df_spodni, f"Pokud {f} <= {hrana}")
                analyzuj_segment(df_horni, f"Pokud {f} > {hrana}")

        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        if vysledky:
            df_vysledky = pd.DataFrame(vysledky)
            # Filtrování falešných 1.0 korelací z malých vzorků
            df_vysledky = df_vysledky[df_vysledky["R²"] < 0.999]
            
            df_vysledky = df_vysledky.drop_duplicates(subset=["R²", "Parametry"])
            df_vysledky = df_vysledky.sort_values(by="R²", ascending=False).reset_index(drop=True)
            
            st.subheader("🏆 Nejlepší nalezené korelace")
            st.dataframe(
                df_vysledky.style.format({"R²": "{:.4f}"}),
                use_container_width=True,
                height=600
            )
            
            st.markdown("### 🔥 Absolutní vítěz:")
            nejlepsi = df_vysledky.iloc[0]
            st.success(f"Ideální hranici nalezl algoritmus zde: **{nejlepsi['Skupina']}**")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Spolehlivost (R²)", f"{nejlepsi['R²']:.4f}")
            with col_b:
                st.metric("Počet vzorků", int(nejlepsi['Vzorků']))
            with col_c:
                st.metric("Ustřelené body odstraněny", int(nejlepsi['Vyhozeno extrémů']))
                
            st.code(nejlepsi['Rovnice pro Excel'], language="excel")
        else:
            st.warning("Algoritmus nenašel žádnou smysluplnou korelaci.")
