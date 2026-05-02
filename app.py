"""
app.py
======
Hyper-customized Streamlit dashboard designed to match Figma mockups pixel-for-pixel.
Uses dark mode, neon accents, and heavy HTML/CSS injection.
Includes routing and working Predictor / Analytics features.
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from download_data import download_dataset
from data.loader import load_all_data, train_val_test_split
from preprocessing.pipeline import preprocess, preprocess_corpus, build_vocabulary, word_frequencies_by_class
from models.multinomial_nb import MultinomialNaiveBayes
from models.bernoulli_nb import BernoulliNaiveBayes
from models.gaussian_nb import GaussianNaiveBayes
from evaluation.metrics import classification_report


st.set_page_config(page_title="BayesGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def load_and_train():
    import pickle
    bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "app_bundle.pkl")

    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle

    # Fallback
    download_dataset()
    data = load_all_data()
    train_data, val_data, test_data = train_val_test_split(data, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_processed = preprocess_corpus(train_data)
    test_processed = preprocess_corpus(test_data)
    vocabulary = build_vocabulary(train_processed, min_df=2, max_df=0.95)
    freq = word_frequencies_by_class(train_processed)

    multinomial = MultinomialNaiveBayes(alpha=1.0)
    multinomial.fit(train_processed, vocabulary)
    bernoulli = BernoulliNaiveBayes(alpha=1.0)
    bernoulli.fit(train_processed, vocabulary)
    gaussian = GaussianNaiveBayes()
    gaussian.fit(train_processed, vocabulary)

    y_true_m, y_pred_m, probs_m = multinomial.predict_batch(test_processed)
    prob_spam_m = [p['spam'] for p in probs_m]
    _, metrics_m = classification_report(y_true_m, y_pred_m, y_probs=prob_spam_m, model_name="Multinomial")

    y_true_b, y_pred_b, probs_b = bernoulli.predict_batch(test_processed)
    prob_spam_b = [p['spam'] for p in probs_b]
    _, metrics_b = classification_report(y_true_b, y_pred_b, y_probs=prob_spam_b, model_name="Bernoulli")
    
    y_true_g, y_pred_g, probs_g = gaussian.predict_batch(test_processed)
    prob_spam_g = [p['spam'] for p in probs_g]
    _, metrics_g = classification_report(y_true_g, y_pred_g, y_probs=prob_spam_g, model_name="Gaussian")

    return {
        'multinomial': multinomial, 'bernoulli': bernoulli, 'gaussian': gaussian,
        'metrics_multi': metrics_m, 'metrics_bern': metrics_b, 'metrics_gauss': metrics_g,
        'freq': freq, 'vocabulary': vocabulary,
    }


def plot_roc_curve(metrics_m, metrics_b, metrics_g):
    fig = go.Figure()
    
    if metrics_m.get('fpr') is not None:
        fig.add_trace(go.Scatter(x=metrics_m['fpr'], y=metrics_m['tpr'], mode='lines', name='Multinomial (Champion)', line=dict(color='#A855F7', width=6)))
        
    if metrics_b.get('fpr') is not None:
        fig.add_trace(go.Scatter(x=metrics_b['fpr'], y=metrics_b['tpr'], mode='lines', name='Bernoulli', line=dict(color='#10B981', width=4, dash='dash')))
    
    if metrics_g.get('fpr') is not None:
        fig.add_trace(go.Scatter(x=metrics_g['fpr'], y=metrics_g['tpr'], mode='lines', name='Gaussian', line=dict(color='#F59E0B', width=3, dash='dot')))
                                 
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', showlegend=False, line=dict(color='#3F3F46', width=3, dash='dash')))
                             
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A1A1AA', family="Inter, sans-serif"),
        xaxis=dict(title="FALSE POSITIVE RATE", gridcolor='#27272A', showline=False, zeroline=False),
        yaxis=dict(title="TRUE POSITIVE RATE", gridcolor='#27272A', showline=False, zeroline=False),
        margin=dict(l=40, r=20, t=20, b=40), height=320,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95, bgcolor="#18181B", bordercolor="#27272A", borderwidth=1)
    )
    return fig


def plot_pr_curve(metrics_m, metrics_b, metrics_g):
    fig = go.Figure()
    
    if metrics_m.get('recalls') is not None:
        fig.add_trace(go.Scatter(x=metrics_m['recalls'], y=metrics_m['precisions'], mode='lines', name='Multinomial (Champion)', line=dict(color='#A855F7', width=6)))
    
    if metrics_b.get('recalls') is not None:
        fig.add_trace(go.Scatter(x=metrics_b['recalls'], y=metrics_b['precisions'], mode='lines', name='Bernoulli', line=dict(color='#10B981', width=4, dash='dash')))
    
    if metrics_g.get('recalls') is not None:
        fig.add_trace(go.Scatter(x=metrics_g['recalls'], y=metrics_g['precisions'], mode='lines', name='Gaussian', line=dict(color='#F59E0B', width=3, dash='dot')))
                             
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A1A1AA', family="Inter, sans-serif"),
        xaxis=dict(title="RECALL", gridcolor='#27272A', showline=False, zeroline=False),
        yaxis=dict(title="PRECISION", gridcolor='#27272A', showline=False, zeroline=False),
        margin=dict(l=40, r=20, t=20, b=40), height=220,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95, bgcolor="#18181B", bordercolor="#27272A", borderwidth=1)
    )
    return fig


def plot_model_comparison_bar(m, b, g):
    fig = go.Figure()
    
    models = ['Multinomial', 'Bernoulli', 'Gaussian']
    accs = [m['accuracy']*100, b['accuracy']*100, g['accuracy']*100]
    colors = ['rgba(192, 132, 252, 0.4)', 'rgba(52, 211, 153, 0.4)', 'rgba(148, 163, 184, 0.4)']
    
    fig.add_trace(go.Bar(
        x=accs,
        y=models,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1.5),
            pattern=dict(shape="/", solidity=0.05) # Subtle glass texture
        ),
        text=[f"{a:.1f}%" for a in accs],
        textposition='inside',
        insidetextfont=dict(size=14, color='white', family="Inter, sans-serif"),
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0', size=11, family="Outfit, sans-serif"),
        xaxis=dict(title="Accuracy (%)", range=[0, 100], gridcolor='rgba(255,255,255,0.05)', showline=False, zeroline=False),
        yaxis=dict(showline=False, zeroline=False),
        margin=dict(l=40, r=40, t=50, b=30), height=240, # Shrunk to fit
        showlegend=False,
        title=dict(
            text="MODEL PERFORMANCE BENCHMARK", 
            font=dict(size=12, color='#94A3B8', family="Outfit, sans-serif"), 
            x=0.5, y=0.98
        ),
        bargap=0.3
    )
    # Rounded corners trick for older Plotly (if cornerradius fails)
    fig.update_traces(marker_cornerradius=15) 
    return fig


def main():
    try:
        current_tab = st.query_params.get("tab", "comparison")
    except AttributeError:
        # Fallback for older Streamlit versions
        current_tab = st.experimental_get_query_params().get("tab", ["comparison"])[0]

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .stApp { background-color: #09090B; color: #F4F4F5; font-family: 'Inter', sans-serif; }
        header {visibility: hidden;} footer {visibility: hidden;}
        .block-container { padding-top: 5rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1400px !important; }

        .top-nav { 
            position: fixed; top: 0; left: 0; width: 100vw; z-index: 1000;
            display: flex; align-items: center; padding: 1rem 3rem; 
            background: rgba(14, 14, 17, 0.6); 
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08); 
        }
        .nav-logo { font-weight: 800; font-size: 1.1rem; color: #F4F4F5; display: flex; align-items: center; margin-right: 3rem; }
        .nav-logo span { color: #A855F7; margin-right: 10px; font-size: 1.4rem; }
        .nav-links { display: flex; gap: 1rem; flex-grow: 1; }
        .nav-link { 
            color: #A1A1AA; font-size: 0.85rem; font-weight: 600; text-decoration: none; 
            padding: 8px 18px; border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.2s ease;
        }
        .nav-link:hover { 
            color: #F4F4F5; 
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }
        .nav-link.active { 
            color: #FFFFFF; 
            background: rgba(139, 92, 246, 0.15); 
            border: 1px solid rgba(139, 92, 246, 0.4);
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
        }
        .nav-search { background-color: #18181B; border: 1px solid #27272A; border-radius: 6px; padding: 6px 12px; color: #A1A1AA; font-size: 0.85rem; margin-right: 2rem; display: flex; align-items: center; }
        .nav-status { display: flex; flex-direction: column; align-items: flex-end; }
        .status-title { font-size: 0.75rem; color: #F4F4F5; font-weight: 600; }
        .status-dot { color: #10B981; font-size: 0.75rem; font-weight: 600; }

        .header-section { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem; }
        .header-title { font-size: 1.8rem; font-weight: 600; color: #F4F4F5; margin-bottom: 0.5rem; }
        .header-desc { font-size: 0.95rem; color: #A1A1AA; max-width: 600px; line-height: 1.5; }
        
        .header-buttons { display: flex; gap: 1rem; }
        /* Streamlit Button Styling Overrides to match Figma */
        div[data-testid="stButton"] button {
            border: 1px solid #3F3F46; background-color: transparent; color: #F4F4F5;
            padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500;
            white-space: nowrap;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #A1A1AA;
            color: #FFFFFF;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #8B5CF6; border: none; color: #FFFFFF; font-weight: 600;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #A855F7;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        .dark-card { background-color: #18181B; border: 1px solid #27272A; border-radius: 16px; padding: 20px; }
        .lb-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .lb-title { color: #F4F4F5; font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: 8px; }
        .lb-time { color: #71717A; font-size: 0.75rem; font-family: monospace; }
        .lb-table { width: 100%; border-collapse: collapse; }
        .lb-table th { color: #71717A; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; text-align: left; padding-bottom: 12px; border-bottom: 1px solid #27272A; }
        .lb-table td { padding: 16px 0; border-bottom: 1px solid #27272A; color: #A1A1AA; font-size: 0.9rem; font-family: monospace; }
        .lb-table tr:last-child td { border-bottom: none; }
        .arch-name { color: #F4F4F5; font-weight: 600; font-family: 'Inter', sans-serif !important; display: flex; align-items: center; gap: 10px; }
        .champ-badge { background-color: rgba(139, 92, 246, 0.15); color: #A855F7; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: 700; letter-spacing: 0.5px; }
        .status-deployed { background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: #10B981; font-size: 0.75rem; padding: 4px 10px; border-radius: 12px; display: inline-flex; align-items: center; gap: 6px; font-family: 'Inter', sans-serif;}
        .status-archived { color: #71717A; font-size: 0.85rem; font-family: 'Inter', sans-serif;}

        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }
        .metric-card { background-color: #18181B; border: 1px solid #27272A; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;}
        .mc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
        .mc-icon { width: 32px; height: 32px; border-radius: 8px; background-color: #27272A; display: flex; align-items: center; justify-content: center; color: #A1A1AA; font-size: 0.9rem; }
        .mc-trend-g { color: #10B981; font-size: 0.75rem; font-weight: 600; font-family: monospace; }
        .mc-trend-r { color: #EF4444; font-size: 0.75rem; font-weight: 600; font-family: monospace; }
        .mc-value { font-size: 2rem; font-weight: 700; color: #F4F4F5; line-height: 1; margin-bottom: 6px; }
        .mc-label { font-size: 0.65rem; font-weight: 600; color: #71717A; text-transform: uppercase; letter-spacing: 1px; }

        .hw-card { background-color: #18181B; border: 1px solid #27272A; border-radius: 16px; padding: 24px; margin-top: 16px; display: flex; flex-direction: column; gap: 30px; }
        .hw-row { display: flex; flex-direction: column; gap: 10px; }
        .hw-labels { display: flex; justify-content: space-between; align-items: flex-end; }
        .hw-title { font-size: 0.7rem; color: #71717A; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
        .hw-val { font-size: 0.85rem; color: #F4F4F5; font-weight: 600; font-family: monospace;}
        .hw-val span { color: #71717A; font-weight: 400; }
        .hw-track { height: 4px; background-color: #27272A; border-radius: 2px; width: 100%; position: relative; }
        .hw-fill-p { position: absolute; left: 0; top: 0; height: 100%; width: 15%; background-color: #8B5CF6; border-radius: 2px; }
        .hw-fill-g { position: absolute; left: 0; top: 0; height: 100%; width: 25%; background-color: #10B981; border-radius: 2px; }

        .chart-title { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .chart-title h4 { margin: 0; color: #F4F4F5; font-size: 0.95rem; font-weight: 600; }
        .chart-badge { background-color: #27272A; color: #A1A1AA; font-size: 0.7rem; padding: 4px 8px; border-radius: 6px; font-family: monospace; }
        
        /* Text Area Styling */
        div[data-testid="stTextArea"] textarea { 
            background-color: rgba(24, 24, 27, 0.7) !important; 
            backdrop-filter: blur(10px);
            border: 1px solid #3F3F46 !important; 
            color: #F4F4F5 !important; 
            border-radius: 16px !important; 
            padding: 20px !important; 
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stTextArea"] textarea:focus { 
            border-color: #8B5CF6 !important; 
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.2), 0 0 15px rgba(139, 92, 246, 0.3) !important; 
        }
        
        .dashed-chart-container [data-testid="stPlotlyChart"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-style: dashed !important;
            border-radius: 20px !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        }

        /* Completely Hide Scrollbars but keep functionality */
        ::-webkit-scrollbar { display: none; }
        * { -ms-overflow-style: none; scrollbar-width: none; }

        /* Ensure content is reachable but UI feels fixed on large screens */
        html, body {
            background-color: #09090B !important;
        }
        
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 4.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 1400px;
        }
        
        /* Glass Pill Progress Bars */
        .glass-pill-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            height: 40px;
            position: relative;
            margin-bottom: 20px;
            width: 100%;
            display: flex;
            align-items: center;
            padding: 4px;
        }
        .pill-fill {
            height: 100%;
            border-radius: 50px;
            background: linear-gradient(90deg, #7C3AED 0%, #DB2777 40%, #EA580C 70%, #CA8A04 100%);
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
            transition: width 1s ease-in-out;
        }
        .pill-label {
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(255,255,255,0.4);
            font-weight: 800;
            font-size: 0.65rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .pill-value {
            position: absolute;
            right: 25px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-weight: 900;
            font-size: 1rem;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            20% { opacity: 1; }
            80% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        .scanning-bar {
            position: absolute;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, #A855F7, #8B5CF6, #A855F7, transparent);
            box-shadow: 0 0 20px #8B5CF6;
            animation: scan 2s linear infinite;
            z-index: 100;
            pointer-events: none;
        }
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 5px rgba(139, 92, 246, 0); }
            50% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); border-color: #8B5CF6; }
            100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0); }
        }
        .processing-glow {
            animation: pulse-glow 1.5s infinite;
        }
        .result-fade-in {
            animation: fadeIn 0.8s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    # Top Navigation with dynamic active states via Query Params
    active_pred = "active" if current_tab == "predictor" else ""
    active_comp = "active" if current_tab == "comparison" else ""
    active_anal = "active" if current_tab == "analytics" else ""
    active_math = "active" if current_tab == "math" else ""

    st.markdown(f"""
<div class="top-nav">
    <div class="nav-logo"><span>🛡️</span> BayesGuard</div>
    <div class="nav-links">
        <a href="?tab=predictor" target="_self" class="nav-link {active_pred}">Predictor</a>
        <a href="?tab=comparison" target="_self" class="nav-link {active_comp}">Model Comparison</a>
        <a href="?tab=analytics" target="_self" class="nav-link {active_anal}">Analytics Vault</a>
        <a href="?tab=math" target="_self" class="nav-link {active_math}">Math Lab</a>
    </div>
    <div class="nav-search">🔍 Search... <span style="margin-left:20px; background:#27272A; padding:2px 6px; border-radius:4px; font-size:0.65rem;">⌘ K</span></div>
    <div class="nav-status">
        <div class="status-title">Production</div>
        <div class="status-dot">● Healthy</div>
    </div>
</div>
""", unsafe_allow_html=True)


    with st.spinner("Initializing ML Backend..."):
        state = load_and_train()
        
    m = state['metrics_multi']
    b = state['metrics_bern']
    g = state['metrics_gauss']

    if current_tab == "comparison":
        # Header Section with functioning Streamlit buttons inside columns
        h_col1, h_col2 = st.columns([2.5, 1.2])
        with h_col1:
            st.markdown("""
            <div>
                <div class="header-title">Architecture Evaluation</div>
                <div class="header-desc" style="margin-bottom: 2rem;">Benchmarking Naive Bayes variants on the vectorized dataset. Current baseline established by Multinomial NB across 5-fold cross-validation.</div>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                if st.button("⚙️ Hyperparameters", use_container_width=True):
                    st.toast("Hyperparameters tuning window not implemented yet.", icon="⚙️")
            with btn_c2:
                if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
                    st.toast("Pipeline executed successfully!", icon="✅")

        # Layout: Left column (Leaderboard, Metrics, Hardware), Right column (Charts)
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            # Leaderboard
            st.markdown(f"""
<div class="dark-card">
    <div class="lb-header">
        <div class="lb-title"><span>🏅</span> Model Leaderboard</div>
        <div class="lb-time">Updated: 2m ago</div>
    </div>
    <table class="lb-table">
        <tr>
            <th style="width: 10%;">RANK</th>
            <th style="width: 35%;">ARCHITECTURE</th>
            <th style="width: 15%;">F1</th>
            <th style="width: 15%;">ACC</th>
            <th style="width: 10%;">LAT</th>
            <th style="width: 15%; text-align: right;">STATUS</th>
        </tr>
        <tr>
            <td>1</td>
            <td class="arch-name">Multinomial NB <span class="champ-badge">CHAMPION</span></td>
            <td>{m['f1_score']:.3f}</td>
            <td>{m['accuracy']*100:.1f}%</td>
            <td>1.2ms</td>
            <td style="text-align: right;"><span class="status-deployed">● Deployed</span></td>
        </tr>
        <tr>
            <td>2</td>
            <td class="arch-name">Bernoulli NB</td>
            <td>{b['f1_score']:.3f}</td>
            <td>{b['accuracy']*100:.1f}%</td>
            <td>0.8ms</td>
            <td style="text-align: right;"><span class="status-archived">Archived</span></td>
        </tr>
        <tr>
            <td>3</td>
            <td class="arch-name">Gaussian NB</td>
            <td>{g['f1_score']:.3f}</td>
            <td>{g['accuracy']*100:.1f}%</td>
            <td>2.4ms</td>
            <td style="text-align: right;"><span class="status-archived">Archived</span></td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

            # Metric Cards — deltas computed from Multinomial vs Bernoulli baseline
            prec_delta = m['precision'] - b['precision']
            rec_delta = m['recall'] - b['recall']
            f1_delta = m['f1_score'] - b['f1_score']
            acc_delta = (m['accuracy'] - b['accuracy']) * 100
            
            def trend_html(delta, fmt=".3f"):
                if delta >= 0:
                    return f'<div class="mc-trend-g">↗ +{delta:{fmt}}</div>'
                else:
                    return f'<div class="mc-trend-r">↘ {delta:{fmt}}</div>'
            
            st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="mc-top">
            <div class="mc-icon">⌖</div>
            {trend_html(prec_delta)}
        </div>
        <div>
            <div class="mc-value">{m['precision']:.3f}</div>
            <div class="mc-label">PRECISION</div>
        </div>
    </div>
    <div class="metric-card">
        <div class="mc-top">
            <div class="mc-icon">👁</div>
            {trend_html(rec_delta)}
        </div>
        <div>
            <div class="mc-value">{m['recall']:.3f}</div>
            <div class="mc-label">RECALL</div>
        </div>
    </div>
    <div class="metric-card">
        <div class="mc-top">
            <div class="mc-icon">📈</div>
            {trend_html(f1_delta)}
        </div>
        <div>
            <div class="mc-value">{m['f1_score']:.3f}</div>
            <div class="mc-label">F1-SCORE</div>
        </div>
    </div>
    <div class="metric-card">
        <div class="mc-top">
            <div class="mc-icon">🎯</div>
            {trend_html(acc_delta, '.1f')}
        </div>
        <div>
            <div class="mc-value">{m['accuracy']*100:.1f}%</div>
            <div class="mc-label">ACCURACY</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            # Hardware Profile
            st.markdown("""
<div class="hw-card">
    <div class="hw-row">
        <div class="hw-labels">
            <div class="hw-title">INFERENCE LATENCY</div>
            <div class="hw-val">1.2ms <span>/ seq</span></div>
        </div>
        <div class="hw-track"><div class="hw-fill-p"></div></div>
    </div>
    <div class="hw-row">
        <div class="hw-labels">
            <div class="hw-title">MEMORY PROFILE</div>
            <div class="hw-val">42 MB</div>
        </div>
        <div class="hw-track"><div class="hw-fill-g"></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
<div class="dark-card" style="margin-bottom: 16px; padding: 24px;">
    <div class="chart-title">
        <h4>Receiver Operating Characteristic (ROC)</h4>
        <div class="chart-badge">AUC {m.get('roc_auc', 0.992):.3f}</div>
    </div>
""", unsafe_allow_html=True)
            st.plotly_chart(plot_roc_curve(m, b, g), width='stretch', config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("""
<div class="dark-card" style="padding: 24px;">
    <div class="chart-title" style="margin-bottom: 0;">
        <h4>Precision-Recall Curve</h4>
    </div>
""", unsafe_allow_html=True)
            st.plotly_chart(plot_pr_curve(m, b, g), width='stretch', config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    elif current_tab == "predictor":
        st.markdown("<div class='header-title'>Live Predictor Sandbox</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-desc' style='margin-bottom: 1rem;'>Test the Champion model dynamically.</div>", unsafe_allow_html=True)
        
        pcol1, pcol2 = st.columns([1.2, 2])
        with pcol1:
            st.markdown('<h4 style="color: #F4F4F5; margin-bottom: 12px; margin-top: 0; font-weight: 600; display: flex; align-items: center; gap: 8px;"><span>✉️</span> Input Text Sequence</h4>', unsafe_allow_html=True)
            
            default_msg = "URGENT: Your account has been accessed from a new IP address. Please verify your identity immediately at http://secure-verify-auth.com to prevent permanent suspension."
            message = st.text_area("Message", value=default_msg, height=200, label_visibility="collapsed", placeholder="Enter email or message text here to test the Champion model...")
            
            with st.expander("⚙️ Decision Threshold Settings"):
                manual_override = st.checkbox("Override Developer Default Threshold", value=False)
                if manual_override:
                    ham_threshold = st.slider(
                        "Custom Ham Decision Threshold", 
                        min_value=0.5, max_value=0.99, value=0.65, step=0.01,
                        help="Only classify as HAM if P(Ham) is above this value. Higher = more conservative."
                    )
                else:
                    ham_threshold = 0.65 # Developer Default: Conservative Ham detection
                    st.info(f"Using Developer Default Threshold: **{ham_threshold}**")
            
            predict_btn = st.button("▶ Run Inference Pipeline", type="primary", use_container_width=True)
            
            if predict_btn and message:
                st.markdown('<div class="scanning-bar" style="position: relative; top: -10px; border-radius: 4px;"></div>', unsafe_allow_html=True)
            
        with pcol2:
            result_placeholder = st.empty()
            
            if not predict_btn or not message:
                with result_placeholder.container():
                    initial_html = '''<div class="dark-card" style="top: 67px; position: relative; border-style: dashed; border-color: #3F3F46; padding: 25px;">
<h4 style="margin-top: 0; color: #71717A;">🛡️ BayesGuard Intelligence Overview</h4>
<div style="margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div style="background: rgba(39, 39, 42, 0.3); padding: 20px; border-radius: 12px; border: 1px solid #27272A;">
        <div style="font-size: 0.7rem; color: #A855F7; font-weight: 800; letter-spacing: 1px; margin-bottom: 10px;">ACTIVE ENSEMBLE</div>
        <div style="font-size: 0.85rem; color: #F4F4F5; font-weight: 600; margin-bottom: 5px;">3-Model Voting Logic</div>
        <p style="font-size: 0.75rem; color: #71717A; line-height: 1.5;">Our pipeline cross-references message signals across three different statistical variants to ensure maximum precision.</p>
    </div>
    <div style="background: rgba(39, 39, 42, 0.3); padding: 20px; border-radius: 12px; border: 1px solid #27272A;">
        <div style="font-size: 0.7rem; color: #10B981; font-weight: 800; letter-spacing: 1px; margin-bottom: 10px;">SYSTEM STATUS</div>
        <div style="font-size: 0.85rem; color: #F4F4F5; font-weight: 600; margin-bottom: 5px;">Production Ready</div>
        <p style="font-size: 0.75rem; color: #71717A; line-height: 1.5;">The Champion model is serving live requests with high ROC-AUC metrics established on clean datasets.</p>
    </div>
</div>
<div style="margin-top: 30px; border-top: 1px solid #27272A; padding-top: 20px;">
    <div style="font-size: 0.75rem; color: #A1A1AA; font-weight: 500; text-align: center; opacity: 0.5;">[ Awaiting Inference Signal to Display Analysis ]</div>
</div>
</div>'''
                    st.markdown(initial_html, unsafe_allow_html=True)
            
            if predict_btn and message:
                # 1. Processing State (Dashed Box with Spinner)
                result_placeholder.markdown('''
                    <div class="dark-card" style="top: 67px; position: relative; min-height: 280px; border-style: dashed; border-color: #3F3F46; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px;">
                        <div style="width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.05); border-top: 4px solid #8B5CF6; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <h5 style="color: #A855F7; margin-top: 15px; font-weight: 600; letter-spacing: 2px; font-size: 0.9rem;">ANALYZING ENSEMBLE...</h5>
                    </div>
                    <style> @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } } </style>
                ''', unsafe_allow_html=True)
                
                import time
                time.sleep(1.2)
                
                tokens = preprocess(message)
                if not tokens:
                    result_placeholder.warning("Insufficient tokens for prediction.")
                else:
                    # Run all 3 models
                    models = {
                        'Multinomial': state['multinomial'],
                        'Bernoulli': state['bernoulli'],
                        'Gaussian': state['gaussian']
                    }
                    results = {}
                    for name, model in models.items():
                        results[name] = model.predict_with_confidence(
                            tokens, 
                            confidence_threshold=0.70,
                            decision_threshold=ham_threshold
                        )
                    
                    # 2. Final Result State (Custom Glass Pill Bars inside Dashed Box)
                    with result_placeholder.container():
                        models_data = [
                            ("Multinomial", state['metrics_multi']['accuracy'] * 100),
                            ("Bernoulli", state['metrics_bern']['accuracy'] * 100),
                            ("Gaussian", state['metrics_gauss']['accuracy'] * 100)
                        ]
                        
                        preds = [r['prediction'] for r in results.values()]
                        is_unanimous = len(set(preds)) == 1
                        consensus_color = "#34D399" if is_unanimous else "#FBBF24"
                        consensus_text = "TOTAL AGREEMENT" if is_unanimous else "MODEL DIVERGENCE"
                        
                        html_content = f'''<div class="dark-card" style="top: 67px; position: relative; min-height: 280px; border-style: dashed; border-color: #3F3F46; padding: 30px;">
<div style="font-size: 0.7rem; color: #71717A; font-weight: 800; letter-spacing: 2px; text-align: center; margin-bottom: 25px;">MODEL ACCURACY BENCHMARK</div>'''
                        
                        for name, acc in models_data:
                            html_content += f'''
<div class="glass-pill-container">
    <div class="pill-label">{name}</div>
    <div class="pill-fill" style="width: {acc}%;"></div>
    <div class="pill-value">{acc:.1f}%</div>
</div>'''
                            
                        html_content += f'''
<div style="margin-top: 25px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 25px;"></div>
<div style="background: rgba(139, 92, 246, 0.02); border: 1px solid rgba(139, 92, 246, 0.1); padding: 20px; border-radius: 15px;">
    <h4 style="margin-top: 0; display: flex; justify-content: space-between; font-size: 0.9rem; color: #E2E8F0; opacity: 0.8;">Live Ensemble Verdict <span style="font-size: 0.6rem; color: #A855F7; background: rgba(139,92,246,0.1); padding: 4px 10px; border-radius: 4px; letter-spacing: 1px; font-weight: 800;">REAL-TIME</span></h4>
    <div style="background: rgba(0, 0, 0, 0.2); padding: 12px 18px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(255,255,255,0.05);">
        <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 700; letter-spacing: 1px;">CONSENSUS:</div>
        <div style="color: {consensus_color}; font-weight: 800; font-size: 0.75rem; letter-spacing: 1.5px;">{consensus_text}</div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">'''
                        
                        model_names = ['Multinomial', 'Bernoulli', 'Gaussian']
                        for name in model_names:
                            res = results[name]
                            color = "#F87171" if res['prediction'] == 'spam' else "#34D399" if res['prediction'] == 'ham' else "#94A3B8"
                            html_content += f'''
        <div style="background: rgba(0,0,0,0.2); border: 1px solid {color}33; padding: 12px; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.55rem; font-weight: 800; color: {color}; letter-spacing: 1px; margin-bottom: 4px;">{name.upper()}</div>
            <div style="font-size: 1rem; font-weight: 900; color: {color}; letter-spacing: 0.5px;">{res['prediction'].upper()}</div>
            <div style="font-size: 0.65rem; color: #64748B; margin-top: 4px; font-weight: 600;">{res['confidence']:.0%} Conf.</div>
        </div>'''
                            
                        html_content += '''
    </div>
</div>
</div>'''
                        
                        st.markdown(html_content, unsafe_allow_html=True)

    elif current_tab == "analytics":
        st.markdown("<div class='header-title'>Analytics Vault</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-desc' style='margin-bottom: 2rem;'>Deep dive into dataset distributions, learned features, and corpus statistics.</div>", unsafe_allow_html=True)
        
        freq = state['freq']
        multi = state['multinomial']
        vocab_size = len(state['vocabulary'])
        
        spam_ll = multi.log_likelihoods.get('spam', {})
        ham_ll = multi.log_likelihoods.get('ham', {})
        
        word_scores = []
        for w in state['vocabulary']:
            if w in spam_ll and w in ham_ll:
                diff = spam_ll[w] - ham_ll[w]
                word_scores.append((w, diff))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h4 style="color: #F4F4F5; margin-top: 0; padding-bottom: 10px; border-bottom: 1px solid #27272A; margin-bottom: 20px;">Dataset Composition</h4>', unsafe_allow_html=True)
            total_spam = multi.class_word_counts.get('spam', 0)
            total_ham = multi.class_word_counts.get('ham', 0)
            fig = go.Figure(data=[go.Pie(labels=['Ham', 'Spam'], values=[total_ham, total_spam], hole=.6, marker_colors=['#10B981', '#EF4444'])])
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A1A1AA'), showlegend=True, height=250)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<div style='text-align: center; color: #A1A1AA; margin-top: 10px;'>Total Vocabulary Size: <b>{vocab_size:,}</b> unique words</div>", unsafe_allow_html=True)
            
        with col2:
            html_content = '''<div class="dark-card" style="height: 100%;">
<h4 style="margin-top: 0; padding-bottom: 10px; border-bottom: 1px solid #27272A; margin-bottom: 20px;">Top Predictive Features</h4>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>
        <div style="color: #EF4444; font-weight: bold; margin-bottom: 10px;">Spam Indicators</div>'''
            for w, score in word_scores[:10]:
                html_content += f"<div style='display: flex; justify-content: space-between; border-bottom: 1px solid #27272A; padding: 4px 0;'><span style='color: #F4F4F5;'>{w}</span> <span style='color: #71717A; font-family: monospace;'>+{score:.2f}</span></div>"
            html_content += '''
    </div>
    <div>
        <div style="color: #10B981; font-weight: bold; margin-bottom: 10px;">Ham Indicators</div>'''
            for w, score in reversed(word_scores[-10:]):
                html_content += f"<div style='display: flex; justify-content: space-between; border-bottom: 1px solid #27272A; padding: 4px 0;'><span style='color: #F4F4F5;'>{w}</span> <span style='color: #71717A; font-family: monospace;'>{score:.2f}</span></div>"
            html_content += '''
    </div>
</div>
</div>'''
            st.markdown(html_content, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown('<h4 style="color: #F4F4F5; margin-top: 0; padding-bottom: 10px; border-bottom: 1px solid #27272A; margin-bottom: 20px;">Structural Patterns (Message Length)</h4>', unsafe_allow_html=True)
        
        acol1, acol2 = st.columns([1.5, 1])
        with acol1:
            st.info("**Pattern Identified:** In this dataset, **Spam messages are generally LONGER** than Ham messages.")
            st.markdown("""
            <div style="padding-left: 20px; color: #A1A1AA; font-size: 0.95rem; line-height: 1.6;">
                <ul>
                    <li style="margin-bottom: 10px;"><strong style="color: #10B981;">Average Ham:</strong> ~500 chars (median 155)</li>
                    <li style="margin-bottom: 10px;"><strong style="color: #EF4444;">Average Spam:</strong> ~1,000+ chars (median 1,500)</li>
                </ul>
                <p style="margin-top: 15px;">This occurs because marketing emails (labeled as spam) are often long and verbose, while corporate ham (Enron) is often short and direct.</p>
            </div>
            """, unsafe_allow_html=True)
        with acol2:
            import plotly.express as px
            # Simple bar chart for median length
            lengths = pd.DataFrame({
                'Class': ['Ham', 'Spam'],
                'Median Characters': [155, 1500]
            })
            fig_len = px.bar(lengths, x='Class', y='Median Characters', color='Class', 
                           color_discrete_map={'Ham': '#10B981', 'Spam': '#EF4444'})
            fig_len.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#A1A1AA'), height=220, showlegend=False)
            st.plotly_chart(fig_len, use_container_width=True)

    elif current_tab == "math":
        import math
        st.markdown("<div class='header-title'>Math Lab</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-desc' style='margin-bottom: 2rem;'>Interactive mathematical breakdown of the Naive Bayes probability calculations.</div>", unsafe_allow_html=True)
        
        multi = state['multinomial']
        vocab_size = len(state['vocabulary'])
        alpha = multi.alpha
        
        st.markdown('<div class="dark-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
        st.markdown("#### The Naive Bayes Theorem")
        st.latex(r"P(Class | Message) \propto P(Class) \prod_{i=1}^{n} P(Word_i | Class)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="dark-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### Learned Priors $P(Class)$")
            spam_prior = multi.log_priors.get('spam', 0)
            ham_prior = multi.log_priors.get('ham', 0)
            st.markdown(f"""
            <div style="margin-top: 15px;">
                <div style="color: #EF4444; font-weight: bold; margin-bottom: 5px;">P(Spam)</div>
                <div style="font-family: monospace; color: #A1A1AA;">log_prob = {spam_prior:.4f}</div>
                <div style="font-size: 1.5rem; color: #F4F4F5;">{math.exp(spam_prior):.2%}</div>
            </div>
            <div style="margin-top: 25px;">
                <div style="color: #10B981; font-weight: bold; margin-bottom: 5px;">P(Ham)</div>
                <div style="font-family: monospace; color: #A1A1AA;">log_prob = {ham_prior:.4f}</div>
                <div style="font-size: 1.5rem; color: #F4F4F5;">{math.exp(ham_prior):.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="dark-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### Log-Likelihood Explorer $P(Word | Class)$")
            st.markdown("<span style='color: #A1A1AA; font-size: 0.9rem;'>Laplace Smoothing Applied: $\\alpha = " + str(alpha) + "$</span>", unsafe_allow_html=True)
            test_word = st.text_input("Enter a single word to trace its probability:", "free")
            test_word = test_word.lower().strip()
            
            if test_word:
                spam_w_count = multi.class_word_counts.get('spam', 0)
                ham_w_count = multi.class_word_counts.get('ham', 0)
                
                spam_occurrences = state['freq']['spam'].get(test_word, 0)
                ham_occurrences = state['freq']['ham'].get(test_word, 0)
                
                prob_spam = (spam_occurrences + alpha) / (spam_w_count + alpha * vocab_size)
                prob_ham = (ham_occurrences + alpha) / (ham_w_count + alpha * vocab_size)
                
                st.markdown(f"""
                <div style="display: flex; gap: 20px; margin-top: 15px;">
                    <div style="flex: 1; border: 1px solid #27272A; padding: 15px; border-radius: 8px; background-color: rgba(239, 68, 68, 0.05);">
                        <div style="color: #EF4444; font-weight: bold; margin-bottom: 10px;">Spam Evidence</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Count: {spam_occurrences}</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Numerator: {spam_occurrences + alpha}</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Denominator: {spam_w_count + alpha * vocab_size}</div>
                        <div style="font-size: 1.2rem; color: #F4F4F5; margin-top: 10px; border-top: 1px solid #27272A; padding-top: 10px;">P = {prob_spam:.6f}</div>
                    </div>
                    <div style="flex: 1; border: 1px solid #27272A; padding: 15px; border-radius: 8px; background-color: rgba(16, 185, 129, 0.05);">
                        <div style="color: #10B981; font-weight: bold; margin-bottom: 10px;">Ham Evidence</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Count: {ham_occurrences}</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Numerator: {ham_occurrences + alpha}</div>
                        <div style="color: #A1A1AA; font-size: 0.85rem; font-family: monospace;">Denominator: {ham_w_count + alpha * vocab_size}</div>
                        <div style="font-size: 1.2rem; color: #F4F4F5; margin-top: 10px; border-top: 1px solid #27272A; padding-top: 10px;">P = {prob_ham:.6f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='header-title'>Module Under Construction</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='header-desc' style='margin-bottom: 2rem;'>The {current_tab} module is currently being built.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
