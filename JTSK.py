import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="Analýza Kb", layout="wide")
st.title("Klasická regrese a korelace (Jako v Excelu)")

uploaded_file = st.file_uploader("Nahraj tvůj soubor (Sešit1.csv)", type=['csv'])

if uploaded_file is not None:
    # 1. Čisté načtení pro tvůj konkrétní formát z Excelu
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='windows-1250')
    
    # 2. Vyčištění a převod na čísla
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna()
    
    # Do bočního panelu dáme možnost oříznout data (protože v Excelu máš 14 řádků, asi jsi filtroval)
    st.sidebar.header("Filtrace dat")
    w_min = float(df['w'].min())
    w_max = float(df['w'].max())
    w_filtr = st.sidebar.slider("Filtrovat vlhkost (w):", w_min, w_max, (w_min, w_max))
    
    # Aplikace filtru
    df = df[(df['w'] >= w_filtr[0]) & (df['w'] <= w_filtr[1])]
    st.success(f"Připraveno: {len(df)} řádků.")

    if len(df) < 3:
        st.error("Příliš málo dat pro výpočet. Uprav filtr.")
        st.stop()

    # 3. Výběr toho, co chceme analyzovat
    cols = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        idx_kb = cols.index('Kb') if 'Kb' in cols else len(cols)-1
        target = st.selectbox("Cílová hodnota (Y):", cols, index=idx_kb)
        
    with col2:
        default_features = [c for c in cols if c != target and c != 'w*w']
        features = st.multiselect("Vstupní hodnoty (X):", cols, default=default_features)

    # 4. Samotná analýza
    if features:
        st.markdown("---")
        X = df[features]
        y = df[target]
        
        # VYTVOŘENÍ KOMBINACÍ (Edef2 * w, w^2, atd.)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        kombinace = poly.get_feature_names_out(X.columns)
        
        # 1. KROK: HLEDÁNÍ NEJSILNĚJŠÍCH KORELACÍ (Pearson)
        vysledky = []
        for i, nazev in enumerate(kombinace):
            sloupec = X_poly[:, i]
            # Kontrola, aby sloupec nebyl samá nula (vyhne se chybám v dělení)
            if sloupec.std() > 0:
                # Pearsonova korelace
                korelace = df[target].corr(pd.Series(sloupec, index=df.index))
                if pd.notna(korelace):
                    vysledky.append({"Matematická vazba": nazev, "Korelace (R)": korelace})
        
        if vysledky:
            df_res = pd.DataFrame(vysledky)
            # Přidáme absolutní hodnotu pro správné seřazení (záporná korelace je taky silná)
            df_res['Abs_R'] = df_res['Korelace (R)'].abs()
            df_res = df_res.sort_values(by='Abs_R', ascending=False)
            
            top_10 = df_res.head(10).drop(columns=['Abs_R'])
            
            # Vykreslení
            fig = px.bar(
                top_10, 
                x='Korelace (R)', 
                y='Matematická vazba', 
                orientation='h', 
                title=f"TOP 10 nejsilnějších vazeb pro {target} (od -1 do 1)",
                color="Korelace (R)",
                color_continuous_scale=px.colors.diverging.RdBu,
                range_color=[-1, 1]
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. KROK: KLASICKÁ REGRESE (KONTROLA S EXCELEM)
            # Vybereme tu úplně nejsilnější vygenerovanou vazbu
            best_feature_name = df_res.iloc[0]['Matematická vazba']
            best_feature_idx = list(kombinace).index(best_feature_name)
            X_best = X_poly[:, best_feature_idx].reshape(-1, 1)
            
            # Prostá lineární regrese (stejná rovnice jako v Analýze dat v Excelu)
            lin_reg = LinearRegression()
            lin_reg.fit(X_best, y)
            y_pred = lin_reg.predict(X_best)
            
            # Výpočet R-kvadrát
            r_squared = r2_score(y, y_pred)
            
            st.markdown("### 📊 Regresní statistika (Srovnání s Excelem)")
            st.write(f"Model automaticky vybral tvou nejsilnější vazbu: **{best_feature_name}**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Hodnota spolehlivosti (R²)", f"{r_squared:.4f}")
            with col_b:
                st.metric("Násobné R (Korelace)", f"{df_res.iloc[0]['Abs_R']:.4f}")
                
            st.info(f"Rovnice pro výpočet: **{target} = {lin_reg.coef_[0]:.4f} * ({best_feature_name}) + {lin_reg.intercept_:.4f}**")
