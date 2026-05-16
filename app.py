"""
All-In-One Streamlit App: Trains on Startup & Predicts
Econometrics Pipeline: Cement Pricing & Cartel Analysis using scikit-learn.

Requirements:
    pip install streamlit pandas numpy scipy scikit-learn openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# This MUST be the very first Streamlit command
st.set_page_config(page_title="Cement Cartel Analyzer", page_icon="🏗️", layout="centered")

# ─────────────────────────────────────────────────────────────────────────────
# 0. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. LIVE TRAINING ENGINE (Runs on site opening)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def train_model():
    """Trains the OLS model and calculates standard econometrics stats."""
    try:
        # Load Data
        df = pd.read_excel('Assignment Data.xlsx').dropna()
        
        # Define Variables
        features = ['FOIL', 'COAL', 'ELECTRICITY', 'GAS', 'PBAG', 'LIMESTONE', 'CLAY', 'GYPSUM', 'GDPGR']
        X = df[features].values.astype(float)
        y = df['CPRICE'].values.astype(float)

        # Fit scikit-learn model
        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)
        
        # Calculate Econometric Statistics (Standard Errors, t-stats, p-values) manually
        predictions = model.predict(X)
        X_matrix = np.append(np.ones((len(X), 1)), X, axis=1) # Add constant
        
        n = len(X)
        k = len(features)
        df_resid = n - (k + 1)
        mse = (sum((y - predictions)**2)) / df_resid
        
        var_b = mse * np.linalg.inv(np.dot(X_matrix.T, X_matrix)).diagonal()
        std_errors = np.sqrt(var_b)
        params = np.append(model.intercept_, model.coef_)
        t_stats = params / std_errors
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), df_resid)) for i in t_stats]
        
        # Map coefficients and p-values
        names = ['const'] + features
        coef_dict = dict(zip(names, params))
        pval_dict = dict(zip(names, p_values))
        
        metrics = {
            'r_squared': float(r_squared),
            'n_obs': int(n)
        }
        return coef_dict, pval_dict, metrics
        
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        return str(e), None, None

# Run training
coefs, pvals, metrics = train_model()

if coefs is None:
    st.error("🚨 `Assignment Data.xlsx` not found! Please place it in the same folder as this script.")
    st.stop()
elif isinstance(coefs, str):
    st.error(f"🚨 Training Error: {coefs}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS & SOCIAL LINKS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📊 Model Diagnostics")
st.sidebar.info(f"**Observations (N):** {metrics['n_obs']}")
st.sidebar.metric(label="R-Squared (Explanatory Power)", value=f"{metrics['r_squared']:.4f}")

if metrics['r_squared'] < 0.50:
    st.sidebar.error("🚨 **High Cartel Risk:** Costs explain less than 50% of cement prices. Strong evidence of arbitrary price hikes.")
elif metrics['r_squared'] < 0.75:
    st.sidebar.warning("⚠️ **Moderate Cartel Risk:** Costs partially explain prices, but significant variation is unexplained.")
else:
    st.sidebar.success("✅ **Producers Innocent:** Production costs explain the vast majority of cement price variations.")

# st.sidebar.markdown("---")
# st.sidebar.markdown("### 👨‍💻 Developer")
# st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Yousef10p-181717?logo=github)](https://github.com/Yousef10p)")
# st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yousef_Alogiely-0A66C2?logo=linkedin)](https://www.linkedin.com/in/yousef-alogiely-29389b283/)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDICTION UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏗️ Cement Price Predictor")
st.write("Input the current market prices for production materials to calculate the justified/expected consumer cement price.")

with st.form("prediction_form"):
    st.markdown("### 🛢️ Fuel & Energy Costs")
    col1, col2, col3 = st.columns(3)
    with col1: foil_val = st.number_input("Furnace Oil (FOIL)", value=2000.0)
    with col2: coal_val = st.number_input("Coal (COAL)", value=35.0)
    with col3: elec_val = st.number_input("Electricity", value=1.5)

    st.markdown("### 🪨 Raw Materials & Misc")
    col4, col5, col6 = st.columns(3)
    with col4: gas_val = st.number_input("Gas", value=16.0)
    with col5: pbag_val = st.number_input("Paper Bag (PBAG)", value=5.0)
    with col6: lime_val = st.number_input("Limestone", value=30.0)

    col7, col8, col9 = st.columns(3)
    with col7: clay_val = st.number_input("Clay", value=25.0)
    with col8: gyp_val = st.number_input("Gypsum", value=130.0)
    with col9: gdp_val = st.number_input("GDP Growth (GDPGR)", value=5.0)

    submitted = st.form_submit_button("🔮 Predict Justified Cement Price (CPRICE)", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    st.session_state.prediction_run = True
    
    user_inputs = {
        'FOIL': foil_val, 'COAL': coal_val, 'ELECTRICITY': elec_val, 
        'GAS': gas_val, 'PBAG': pbag_val, 'LIMESTONE': lime_val, 
        'CLAY': clay_val, 'GYPSUM': gyp_val, 'GDPGR': gdp_val
    }

    base_const = coefs.get('const', 0.0)
    expected_cprice = base_const
    debug_data = [{'Variable': 'Intercept (const)', 'User Input': 1.0, 'Coefficient': base_const, 'P-Value': pvals['const'], 'Contribution': base_const}]

    for var_name, value in user_inputs.items():
        if var_name in coefs:
            coef = coefs[var_name]
            pval = pvals[var_name]
            contribution = coef * value
            expected_cprice += contribution
            debug_data.append({
                'Variable': var_name, 
                'User Input': round(value, 4), 
                'Coefficient': round(coef, 6), 
                'P-Value': round(pval, 4),
                'Contribution': round(contribution, 4)
            })

    st.divider()
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 2px solid #4CAF50;">
            <h2 style="margin:0;">Expected Cement Price (Based on Costs)</h2>
            <h1 style="color: #4CAF50; font-size: 4rem; margin: 10px 0;">${expected_cprice:.2f}</h1>
            <h3 style="margin:0; color: #aaaaaa;">If the actual market price is much higher, suspect cartelization.</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🛠️ View Econometrics Math Breakdown"):
        st.write(f"**Overall Explanatory Power ($R^2$):** `{metrics['r_squared']:.4f}`")
        st.latex(r"CPRICE = \beta_0 + \beta_1FOIL + \beta_2COAL + ... + \epsilon")
        st.dataframe(pd.DataFrame(debug_data).style.apply(
            lambda x: ['background: #ffe6e6' if v > 0.05 else 'background: #e6ffe6' for v in x], 
            subset=['P-Value']
        ), use_container_width=True)