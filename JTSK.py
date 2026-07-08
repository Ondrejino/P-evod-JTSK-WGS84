import streamlit as st
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="AI Analýza & Extrémy", layout="wide")
st.title("🤖 Inteligentní hledání korelací a extrémů")
st.markdown("Automaticky zkouší dynamické řezy v datech a detekuje hodnoty, které kazí korelaci.")

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
        features = st.multiselect("Vstupní hodnoty (X):", cols, default=default_features)

    if features:
        st.markdown("---")
        
        vysledky = []
        
        # Funkce pro výpočet regrese vč. detekce extrémů
        def analyzuj_segment(df_vzorek, nazev_segmentu):
            # Musí zbýt aspoň 6 řádků, aby mělo smysl něco vyhazovat a počítat
            if len(df_vzorek) < 6:
                return
                
            y = df_vzorek[target]
            
            for pocet_promennych in range(1, len(features) + 1):
                for kombinace in itertools.combinations(features, pocet_promennych):
                    X_subset = df_vzorek[list(kombinace)]
                    
                    if X_subset.std().min() == 0:
                        continue
                        
                    # 1. KROK: Výpočet se všemi daty v segmentu
                    model = LinearRegression()
                    model.fit(X_subset, y)
                    y_pred = model.predict(X_subset)
                    r2_puvodni = r2_score(y, y_pred)
                    
                    casti = [f"{k:+.4f}*{n}" for k, n in zip(model.coef_, kombinace)]
                    rovnice = f"={model.intercept_:.4f} " + " ".join(casti)
                    
                    vysledky.append({
                        "Skupina": nazev_segmentu,
                        "R² (Spolehlivost)": r2_puvodni,
                        "Odstraněno extrémů": 0,
                        "Počet dat": len(df_vzorek),
                        "Parametry": ", ".join(kombinace),
                        "Rovnice pro Excel": rovnice
                    })
                    
                    # 2. KROK: Hledání a odstranění extrémů (Outliers)
                    rezidua = np.abs(y - y_pred)
                    odchylka = rezidua.std()
                    # Identifikace bodů, které ustřelily o více než 1.5x průměrnou chybu
                    outlier_mask = rezidua > (1.5 * odchylka)
                    pocet_outlieru = outlier_mask.sum()
                    
                    # Pokud se našlo 1 až 3 extrémy a po smazání nám zbydou aspoň 4 body
                    if 0 < pocet_outlieru <= 3 and (len(df_vzorek) - pocet_outlieru) >= 4:
                        df_ciste = df_vzorek[~outlier_mask]
                        X_ciste = df_ciste[list(kombinace)]
                        y_ciste = df_ciste[target]
                        
                        model_cisty = LinearRegression()
                        model_cisty.fit(X_ciste, y_ciste)
                        y_pred_cisty = model_cisty.predict(X_ciste)
                        r2_cisty = r2_score(y_ciste, y_pred_cisty)
                        
                        # Zapsat jen tehdy, pokud to vyhození reálně zlepšilo model
                        if r2_cisty > r2_puvodni:
                            casti_ciste = [f"{k:+.4f}*{n}" for k, n in zip(model_cisty.coef_, kombinace)]
                            rovnice_ciste = f"={model_cisty.intercept_:.4f} " + " ".join(casti_ciste)
                            
                            vysledky.append({
                                "Skupina": f"{nazev_segmentu} (BEZ EXTRÉMŮ)",
                                "R² (Spolehlivost)": r2_cisty,
                                "Odstraněno extrémů": pocet_outlieru,
                                "Počet dat": len(df_ciste),
                                "Parametry": ", ".join(kombinace),
                                "Rovnice pro Excel": rovnice_ciste
                            })

        # --- 1. Zkoumáme celek ---
        analyzuj_segment(df, "Celý dataset (Všechna data)")

        # --- 2. Zkoumáme dynamické rozdělení (33%, 50%, 66%) ---
        percentily = [33, 50, 66]
        
        for f in features:
            for p in percentily:
                hrana = np.percentile(df[f], p)
                
                df_spodni = df[df[f] <= hrana]
                df_horni = df[df[f] > hrana]
                
                # Aby nám nezkoumal zbytečně duplicity, pokud percentil trefí to samé číslo
                analyzuj_segment(df_spodni, f"Když {f} <= {hrana:.2f} (Spodní {p} %)")
                if p != 50: # (Horní část u 50% je stejná, stačí pípnutí na 66% a 33%)
                    analyzuj_segment(df_horni, f"Když {f} > {hrana:.2f} (Horní {100-p} %)")

        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        if vysledky:
            df_vysledky = pd.DataFrame(vysledky)
            # Filtrujeme dokonalé nesmysly a řadíme
            df_vysledky = df_vysledky[df_vysledky["R² (Spolehlivost)"] < 1.0] 
            
            # Odstraníme úplné duplikáty výsledků, pokud se řezal dataset přesně na stejném místě
            df_vysledky = df_vysledky.drop_duplicates(subset=["R² (Spolehlivost)", "Parametry"])
            
            df_vysledky = df_vysledky.sort_values(by="R² (Spolehlivost)", ascending=False).reset_index(drop=True)
            
            st.subheader("🏆 Nalezené korelace (vč. detekce odlehlých hodnot)")
            st.dataframe(
                df_vysledky.style.format({"R² (Spolehlivost)": "{:.4f}"}),
                use_container_width=True,
                height=600
            )
            
            st.markdown("### 🔥 Nejluxusnější nalezená vazba:")
            nejlepsi = df_vysledky.iloc[0]
            st.success(f"Absolutně nejlepší výsledek je ve skupině: **{nejlepsi['Skupina']}**")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Spolehlivost (R²)", f"{nejlepsi['R² (Spolehlivost)']:.4f}")
            with col_b:
                st.metric("Analyzováno vzorků", int(nejlepsi['Počet dat']))
            with col_c:
                st.metric("Zahozeno extrémů", int(nejlepsi['Odstraněno extrémů']))
                
            st.code(nejlepsi['Rovnice pro Excel'], language="excel")
