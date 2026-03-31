"""
app.py
======
Streamlit web application for interactive spam classification.

Features:
    - Text input for entering a message
    - Model selector (Multinomial / Bernoulli)
    - Real-time prediction with probability display
    - Model comparison dashboard
    - Visualization gallery

Usage:
    streamlit run app.py
"""

import os
import sys

# Force UTF-8 output on Windows to avoid encoding errors with Unicode characters
# This handles the Greek letters (α, Σ) and box-drawing chars when Streamlit captures stdout
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from download_data import download_dataset
from data.loader import load_all_data, train_test_split
from preprocessing.pipeline import (
    preprocess, preprocess_corpus, build_vocabulary,
    word_frequencies_by_class
)
from models.multinomial_nb import MultinomialNaiveBayes
from models.bernoulli_nb import BernoulliNaiveBayes
from evaluation.metrics import classification_report, confusion_matrix


# ─── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Spam Classifier — Naïve Bayes",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_and_train():
    """
    Load data and train both models.
    Cached so it only runs once per session.
    """
    download_dataset()

    # Load ALL available datasets (SMS + SpamAssassin)
    data = load_all_data()
    train_data, test_data = train_test_split(data, test_ratio=0.2, seed=42)

    # Preprocess
    train_processed = preprocess_corpus(train_data)
    test_processed = preprocess_corpus(test_data)
    vocabulary = build_vocabulary(train_processed)

    # Word frequencies
    freq = word_frequencies_by_class(train_processed)

    # Train Multinomial NB
    multinomial = MultinomialNaiveBayes(alpha=1.0)
    multinomial.fit(train_processed, vocabulary)

    # Train Bernoulli NB
    bernoulli = BernoulliNaiveBayes(alpha=1.0)
    bernoulli.fit(train_processed, vocabulary)

    # Evaluate both on test set
    y_true_m, y_pred_m, _ = multinomial.predict_batch(test_processed)
    _, metrics_m = classification_report(y_true_m, y_pred_m, "Multinomial")

    y_true_b, y_pred_b, _ = bernoulli.predict_batch(test_processed)
    _, metrics_b = classification_report(y_true_b, y_pred_b, "Bernoulli")

    return {
        'multinomial': multinomial,
        'bernoulli': bernoulli,
        'data': data,
        'train_data': train_data,
        'test_data': test_data,
        'metrics_multi': metrics_m,
        'metrics_bern': metrics_b,
        'freq': freq,
        'vocabulary': vocabulary,
    }


def main():
    """Main Streamlit app."""

    # ─── Custom CSS ──────────────────────────────────────────────
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .spam-box {
            background-color: #ffe6e6;
            border: 2px solid #e74c3c;
            color: #c0392b;
        }
        .ham-box {
            background-color: #e6ffe6;
            border: 2px solid #27ae60;
            color: #1e8449;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
    </style>
    """, unsafe_allow_html=True)

    # ─── Header ──────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📧 Probabilistic Spam Classifier</h1>
        <p style="font-size: 1.1rem; color: #666;">
            Using Naïve Bayes — Built from Scratch with Bayes' Theorem
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Load Models ─────────────────────────────────────────────
    with st.spinner("Loading dataset and training models..."):
        state = load_and_train()

    # ─── Sidebar ─────────────────────────────────────────────────
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.radio(
        "Select Model:",
        ["Multinomial NB", "Bernoulli NB"],
        index=0,
        help="**Multinomial**: Uses word frequency counts\n\n"
             "**Bernoulli**: Uses binary word presence (0/1)"
    )

    model = (state['multinomial'] if model_choice == "Multinomial NB"
             else state['bernoulli'])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.markdown(f"- **Total messages:** {len(state['data'])}")
    st.sidebar.markdown(f"- **Training:** {len(state['train_data'])}")
    st.sidebar.markdown(f"- **Testing:** {len(state['test_data'])}")
    st.sidebar.markdown(f"- **Vocabulary:** {len(state['vocabulary'])} words")

    # ─── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Predict", "📊 Model Comparison",
        "📈 Visualizations", "📐 Mathematical Details"
    ])

    # ─── Tab 1: Prediction ───────────────────────────────────────
    with tab1:
        st.subheader(f"Classify a Message ({model_choice})")

        col1, col2 = st.columns([3, 1])

        with col1:
            message = st.text_area(
                "Enter a message to classify:",
                height=120,
                placeholder="Type or paste a message here...\n"
                            "e.g., 'Congratulations! You won a free iPhone!'"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 Predict", type="primary",
                                     use_container_width=True)

            # Quick test buttons
            st.markdown("**Quick Tests:**")
            if st.button("📩 Spam Example", use_container_width=True):
                message = "WINNER! You've been selected for a FREE cruise! Call 1-800-CRUISE now to claim your prize!"
                predict_btn = True
            if st.button("✉️ Ham Example", use_container_width=True):
                message = "Hey, are you free for lunch tomorrow? Let me know!"
                predict_btn = True

        if predict_btn and message:
            tokens = preprocess(message)

            if not tokens:
                st.warning("After preprocessing, no meaningful words remain. "
                           "Try a longer message.")
            else:
                # Use confidence-based prediction for ambiguity handling
                result = model.predict_with_confidence(tokens, confidence_threshold=0.70)
                pred = result['prediction']
                probs = result['probabilities']

                # Display result
                if pred == 'spam':
                    st.markdown(
                        '<div class="prediction-box spam-box">'
                        'SPAM</div>',
                        unsafe_allow_html=True
                    )
                elif pred == 'uncertain':
                    st.markdown(
                        '<div class="prediction-box" style="background-color: #fff3cd; '
                        'border: 2px solid #ffc107; color: #856404; padding: 1.5rem; '
                        'border-radius: 12px; text-align: center; font-size: 1.3rem; '
                        'font-weight: bold; margin: 1rem 0;">'
                        'UNCERTAIN - Ambiguous Message</div>',
                        unsafe_allow_html=True
                    )
                    st.info(
                        "This message is borderline. Different people might "
                        "classify it differently. The model's confidence is "
                        f"below the 70% threshold ({result['confidence']:.1%})."
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box ham-box">'
                        'HAM (Not Spam)</div>',
                        unsafe_allow_html=True
                    )

                # Probability bars
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("P(Spam | Message)", f"{probs['spam']:.6f}")
                    st.progress(probs['spam'])
                with col_p2:
                    st.metric("P(Ham | Message)", f"{probs['ham']:.6f}")
                    st.progress(probs['ham'])

                # Confidence indicator
                st.markdown(f"**Confidence:** {result['confidence']:.4f} | "
                           f"**Confident:** {'Yes' if result['is_confident'] else 'No'}")

                # Show preprocessing details
                with st.expander("Preprocessing Details"):
                    st.markdown(f"**Original:** {message}")
                    st.markdown(f"**Tokens:** {tokens}")
                    st.markdown(f"**Model:** {model_choice}")
                    st.markdown(f"**Smoothing alpha:** {model.alpha}")
                    st.markdown(f"**Explanation:** {result['explanation']}")

    # ─── Tab 2: Model Comparison ─────────────────────────────────
    with tab2:
        st.subheader("Multinomial vs Bernoulli Naïve Bayes")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Multinomial NB")
            st.markdown("*Based on word frequency counts*")
            m = state['metrics_multi']
            st.metric("Accuracy", f"{m['accuracy']:.4f}")
            st.metric("Precision", f"{m['precision']:.4f}")
            st.metric("Recall", f"{m['recall']:.4f}")
            st.metric("F1 Score", f"{m['f1_score']:.4f}")

        with col2:
            st.markdown("### Bernoulli NB")
            st.markdown("*Based on binary word presence*")
            m = state['metrics_bern']
            st.metric("Accuracy", f"{m['accuracy']:.4f}")
            st.metric("Precision", f"{m['precision']:.4f}")
            st.metric("Recall", f"{m['recall']:.4f}")
            st.metric("F1 Score", f"{m['f1_score']:.4f}")

        st.markdown("---")

        # Comparison table
        st.markdown("### Detailed Comparison Table")
        import pandas as pd
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Multinomial NB': [
                f"{state['metrics_multi'][k]:.4f}"
                for k in ['accuracy', 'precision', 'recall', 'f1_score']
            ],
            'Bernoulli NB': [
                f"{state['metrics_bern'][k]:.4f}"
                for k in ['accuracy', 'precision', 'recall', 'f1_score']
            ],
        }
        df = pd.DataFrame(comparison_data)
        st.table(df)

        # Top words comparison
        st.markdown("### Top 10 Spam vs Ham Words")
        col_s, col_h = st.columns(2)
        with col_s:
            st.markdown("**🔴 Top Spam Words:**")
            for word, count in state['freq']['spam'].most_common(10):
                st.markdown(f"- `{word}` — {count}")
        with col_h:
            st.markdown("**🟢 Top Ham Words:**")
            for word, count in state['freq']['ham'].most_common(10):
                st.markdown(f"- `{word}` — {count}")

    # ─── Tab 3: Visualizations ───────────────────────────────────
    with tab3:
        st.subheader("Generated Plots")

        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs"
        )

        plot_files = [
            ("Word Frequency Distribution", "word_frequency.png"),
            ("Word Clouds: Spam vs Ham", "word_clouds.png"),
            ("Accuracy Comparison", "accuracy_comparison.png"),
            ("Confusion Matrix — Multinomial", "confusion_matrix_multinomial.png"),
            ("Confusion Matrix — Bernoulli", "confusion_matrix_bernoulli.png"),
            ("Error Distribution", "error_distribution.png"),
            ("Cross Validation Results", "cv_results.png"),
        ]

        found_any = False
        for title, filename in plot_files:
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                found_any = True
                st.markdown(f"#### {title}")
                st.image(path, use_container_width=True)
                st.markdown("---")

        if not found_any:
            st.info("📊 No plots found yet. Run `python main.py` first to "
                    "generate all visualizations, then refresh this page.")

    # ─── Tab 4: Math Details ─────────────────────────────────────
    with tab4:
        st.subheader("Mathematical Foundation")

        st.markdown("""
        ### Bayes' Theorem

        The classifier works by computing:

        $$P(\\text{class} | \\text{message}) = \\frac{P(\\text{message} | \\text{class}) \\times P(\\text{class})}{P(\\text{message})}$$

        Since $P(\\text{message})$ is constant for all classes:

        $$P(\\text{class} | \\text{message}) \\propto P(\\text{class}) \\times P(\\text{message} | \\text{class})$$

        ### Naïve Independence Assumption

        We assume words are **conditionally independent** given the class:

        $$P(\\text{message} | \\text{class}) = \\prod_{i=1}^{n} P(w_i | \\text{class})$$

        ### Log Transformation

        To avoid numerical underflow (multiplying many small numbers):

        $$\\log P(\\text{class} | \\text{msg}) = \\log P(\\text{class}) + \\sum_{i=1}^{n} \\log P(w_i | \\text{class})$$

        ### Laplace Smoothing

        To handle unseen words ($P = 0$ problem):

        $$P(w | c) = \\frac{\\text{count}(w, c) + \\alpha}{N_c + \\alpha \\times |V|}$$

        where $\\alpha = 1$ (Laplace smoothing), $N_c$ = total words in class, $|V|$ = vocabulary size .

        ---

        ### Multinomial vs Bernoulli

        | Feature | Multinomial | Bernoulli |
        |---------|-------------|-----------|
        | Word representation | Frequency (count) | Presence (0/1) |
        | P(word\\|class) denominator | $N_c + \\alpha|V|$ | $N_{docs,c} + 2\\alpha$ |
        | Absent words | Ignored | Penalized with $1-P(w|c)$ |
        | Best for | Longer texts | Short texts |
        """)


if __name__ == "__main__":
    main()
