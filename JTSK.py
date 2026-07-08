import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
import plotly.express as px

st.set_page_config(page_title="Analýza Kb", layout="wide")
st.title("Analýza závislostí pro Kb (Celý dataset)")

uploaded_file = st.file_uploader("Nahraj tvůj soubor (Sešit1.csv)", type=['csv'])

if uploaded_file is not None:
    # 1. Čisté načtení pro tvůj konkrétní formát z Excelu
    try:
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='windows-1250')
    
    # 2. Rychlé vyčištění
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0']) # Zahodí prázdný sloupec s názvy STA01...
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna()
    st.success(f"Úspěšně načteno všech {len(df)} řádků z tvého souboru.")
    
    # 3. Výběr toho, co chceme analyzovat
    cols = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        # Cílová hodnota (automaticky se pokusí najít Kb)
        idx_kb = cols.index('Kb') if 'Kb' in cols else len(cols)-1
        target = st.selectbox("Co chceme vypočítat (Cíl):", cols, index=idx_kb)
        
    with col2:
        # Z čeho to chceme počítat (automaticky vybere zbytek)
        default_features = [c for c in cols if c != target and c != 'w*w']
        features = st.multiselect("Z jakých hodnot to chceme počítat (Vstupy):", cols, default=default_features)

    # 4. Samotná analýza (spustí se jen když jsou vybrané vstupy)
    if features:
        st.markdown("---")
        X = df[features]
        y = df[target]
        
        # Vygenerování mocnin a násobků (Edef2 * Evd, vlhkost^2 atd.)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        kombinace = poly.get_feature_names_out(X.columns)
        
        # Srovnání vah (aby tisíce u objemové hmotnosti nepřebily desítky u Evd)
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)
        
        # Lasso Regrese (matematické síto)
        lasso = LassoCV(cv=2, random_state=42, max_iter=10000)
        lasso.fit(X_poly_scaled, y)
        
        # Vytažení jen těch rovnic, které mají smysl
        vysledky = [{"Matematická vazba": k, "Síla vlivu": v} for k, v in zip(kombinace, lasso.coef_) if abs(v) > 0.0001]
        
        if vysledky:
            df_res = pd.DataFrame(vysledky)
            df_res['Absolutní síla'] = df_res['Síla vlivu'].abs()
            df_res = df_res.sort_values(by='Absolutní síla', ascending=False).head(10)
            
            # Vykreslení
            fig = px.bar(
                df_res, 
                x='Síla vlivu', 
                y='Matematická vazba', 
                orientation='h', 
                title=f"TOP 10 faktorů ovlivňujících {target}",
                color="Síla vlivu",
                color_continuous_scale=px.colors.diverging.RdBu
            )
            # Nejsilnější nahoru
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Model nenašel v těchto datech žádnou silnou závislost.")
