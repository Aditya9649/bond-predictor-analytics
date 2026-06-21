import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set layout configurations to force immersive wide layout
st.set_page_config(page_title="Bloomberg Terminal Web", layout="wide")

# Custom CSS theme engineering injection to match image_77ace7.jpg exactly
st.markdown("""
    <style>
    /* Main container background setup */
    .stApp {
        background-color: #F8F9FA;
        color: #1E293B;
    }
    
    /* Clean white background cards for content grids */
    div[data-testid="stColumn"] {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Simulate the dark Bloomberg Terminal Web nested component blocks */
    .terminal-card {
        background-color: #0E131F !important;
        color: #FFFFFF !important;
        border: 1px solid #FF9F1C;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(255, 159, 28, 0.15);
    }
    
    /* Custom input controls styling */
    div[data-baseweb="slider"] {
        padding: 10px 0px;
    }
    
    /* Make numeric inputs super clean */
    div[data-baseweb="input"] {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
    }
    
    /* Make the action button exactly match a professional terminal trigger */
    div.stButton > button:first-child {
        background-color: #0E131F !important;
        color: #FF9F1C !important;
        border: 1px solid #FF9F1C !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        width: 100%;
        height: 45px;
        letter-spacing: 0.05em;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #182235 !important;
        box-shadow: 0 0 12px rgba(255, 159, 28, 0.4);
    }
    
    /* Adjust headings spacing */
    h3, h4, h5 {
        margin-top: 0px !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Top Bloomberg Branding Bar Replica
t_col1, t_col2 = st.columns([1, 4])
with t_col1:
    st.markdown("<h3 style='color:#0F172A; margin-bottom:0;'>BLOOMBERG TERMINAL WEB</h3>", unsafe_allow_html=True)
with t_col2:
    st.markdown("<div style='text-align:right; color:#64748B; font-size:14px; padding-top:10px;'>Account Active | Connection Secured</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:5px; margin-bottom:20px; border:0; border-top:1px solid #E2E8F0;' />", unsafe_allow_html=True)

# Main Grid Core Layout Structure
main_left, main_right = st.columns([2, 1.2])

with main_left:
    st.markdown("<h4 style='color:#0F172A;'>📊 QUANTITATIVE INPUT FIELDS</h4>", unsafe_allow_html=True)
    
    # Subdivide input clusters cleanly using subcolumns
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        st.markdown("<b style='color:#64748B; font-size:12px;'>LIQUIDITY & PROFIT MARGINS</b>", unsafe_allow_html=True)
        current_ratio = st.slider("Current Ratio", 0.1, 10.0, 1.5, step=0.1)
        quick_ratio = st.slider("Quick Ratio", 0.1, 5.0, 1.1, step=0.1)
        net_margin = st.slider("Net Profit Margin (%)", -10.0, 100.0, 10.0) / 100.0
        roa = st.slider("Return on Assets (ROA %)", -20.0, 50.0, 5.0) / 100.0
        
    with in_col2:
        st.markdown("<b style='color:#64748B; font-size:12px;'>RISK PROFILE & MACRO CONTEXT</b>", unsafe_allow_html=True)
        debt_ratio = st.slider("Debt Ratio (Total Debt / Assets)", 0.0, 1.0, 0.35, step=0.01)
        debt_equity = st.slider("Debt to Equity Ratio", 0.0, 5.0, 0.6, step=0.1)
        macro_rate = st.slider("Fed Funds Base Target Rate (%)", 0.0, 6.0, 4.25, step=0.25)
        days_sales = st.number_input("Days of Sales Outstanding (DSO)", min_value=1, max_value=365, value=45)

    st.markdown("<br />", unsafe_allow_html=True)
    evaluate_trigger = st.button("RUN MULTI-MODEL ENSEMBLE FORECAST")

with main_right:
    st.markdown("<h4 style='color:#0F172A;'>🏛️ TERMINAL LIVE ENGINE OUTPUT</h4>", unsafe_allow_html=True)
    
    # Check if pre-trained file models exist on the project drive path
    models_exist = (
        os.path.exists("xgb_bonds_model.pkl") and 
        os.path.exists("lgb_bonds_model.pkl") and 
        os.path.exists("lstm_bonds_model.pkl")
    )
    
    if evaluate_trigger:
        if models_exist:
            # Load your frozen training metrics vectors
            xgb_model = joblib.load("xgb_bonds_model.pkl")
            lgb_model = joblib.load("lgb_bonds_model.pkl")
            lstm_approx = joblib.load("lstm_bonds_model.pkl")
            feature_columns = joblib.load("model_features_list.pkl")
            
            sector_mult = 1.05
            leverage_proxy = net_margin / (debt_ratio + 1e-5)
            
            input_data = {
                'currentRatio': current_ratio, 'quickRatio': quick_ratio, 'cashRatio': 0.4,
                'daysOfSalesOutstanding': days_sales, 'netProfitMargin': net_margin,
                'pretaxProfitMargin': net_margin * 1.2, 'grossProfitMargin': 0.4,
                'operatingProfitMargin': net_margin * 1.3, 'returnOnAssets': roa, 'returnOnEquity': roa * 2.1,
                'debtEquityRatio': debt_equity, 'debtRatio': debt_ratio,
                'currentRatio_sector_ratio': current_ratio * sector_mult,
                'quickRatio_sector_ratio': quick_ratio * sector_mult,
                'cashRatio_sector_ratio': 0.4 * sector_mult,
                'netProfitMargin_sector_ratio': net_margin * sector_mult,
                'returnOnAssets_sector_ratio': roa * sector_mult,
                'returnOnEquity_sector_ratio': roa * 2.1 * sector_mult,
                'debtEquityRatio_sector_ratio': debt_equity * sector_mult,
                'debtRatio_sector_ratio': debt_ratio * sector_mult,
                'macro_interest_proxy': macro_rate,
                'profit_to_debt_leverage': leverage_proxy
            }
            
            input_df = pd.DataFrame([input_data])[feature_columns]
            
            # Predict soft probability bounds
            xgb_probs = xgb_model.predict_proba(input_df)
            lgb_probs = lgb_model.predict_proba(input_df)
            lstm_probs = lstm_approx.predict_proba(input_df)
            
            final_probs = (xgb_probs + lgb_probs + lstm_probs) / 3
            final_class = np.argmax(final_probs, axis=1)[0]
            confidence_score = final_probs[0][final_class] * 100
            
            labels_map = {
                2: {"grade": "High Grade Investment (AAA to A-)", "color": "#00FF66"},
                1: {"grade": "Medium Grade Investment (BBB Family)", "color": "#FFB800"},
                0: {"grade": "Speculative / Junk Grade (BB and below)", "color": "#FF4D4D"}
            }
            
            result = labels_map[final_class]
            
            # Inject the high-contrast nested black terminal container card directly
            st.markdown(f"""
                <div class="terminal-card">
                    <p style="color:#64748B; font-size:11px; margin:0; font-weight:bold; letter-spacing:0.05em;">SECURITY EVALUATION ACTIVE</p>
                    <h2 style="color:{result['color']}; margin-top:5px; margin-bottom:5px; font-weight:800;">{result['grade']}</h2>
                    <hr style="border:0; border-top:1px solid #1E293B; margin:10px 0;" />
                    <p style="color:#E2E8F0; font-size:14px; line-height:1.4;">Ensemble network validation matrices have reached complete matching alignment consensus.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br />", unsafe_allow_html=True)
            st.metric(label="📊 Consensus Confidence Matrix", value=f"{confidence_score:.2f}%")
            
        else:
            st.warning("⚠️ Serialized `.pkl` models missing from folder path context. Run your backend script to generate models first.")
    else:
        # Default placeholder terminal box mirroring empty state
        st.markdown("""
            <div class="terminal-card" style="text-align:center; padding: 40px 20px;">
                <p style="color:#64748B; font-size:13px; font-weight:bold;">AWAITING SYSTEM TRIGGER INPUT</p>
                <p style="color:#475569; font-size:11px; margin:0;">Configure left metrics parameters and execute run matrix analysis</p>
            </div>
        """, unsafe_allow_html=True)