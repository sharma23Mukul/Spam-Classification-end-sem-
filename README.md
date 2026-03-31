# Probabilistic Spam Classification Using Naïve Bayes

A complete end-to-end spam classification pipeline built **entirely from scratch** focusing on core probability and statistics concepts, bypassing black-box machine learning libraries.

## 🚀 Features
- **Multi-Dataset Training**: Trains on **10,128 messages** from both the SMS Spam Collection (UCI) and the SpamAssassin Email Corpus for robust, multi-domain classification.
- **Implementations from Scratch**: Both **Multinomial Naïve Bayes** and **Bernoulli Naïve Bayes** models are implemented using NumPy and math libraries only, relying directly on Bayes' Theorem.
- **Label Ambiguity Handling**: Instead of forcing a binary classification, the models use a **Confidence Threshold (70%)**. Ambiguous or borderline messages are flagged as "UNCERTAIN" to reflect real-world label noise.
- **Statistical Evaluation Suite**: Includes comprehensive analysis (Confusion Matrices, Precision/Recall, 5-Fold Cross Validation, 95% Confidence Intervals, and Hypothesis Testing via paired t-test and McNemar's test).
- **Interactive UI**: A Streamlit dashboard (`app.py`) for real-time inference, model comparison, and mathematical visualizations.

## ⚙️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/probability-spam-classifier.git
cd probability-spam-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the full analysis pipeline** 
The script will automatically download the datasets, train the models, run statistical tests, and generate visualizations in the `outputs/` folder.
```bash
python main.py
```

4. **Launch the interactive Streamlit App**
```bash
streamlit run app.py
```

## 📊 Dataset Specifications
- **SMS Spam Collection**: 5,574 SMS messages (Almeida & Gomez Hidalgo, 2012)
- **SpamAssassin Public Corpus**: ~4,554 email messages (Apache)
- **Total**: 10,128 messages

## 🧪 Statistical Results
- **Multinomial NB Accuracy**: 96.30%
- **Bernoulli NB Accuracy**: 89.28%
- Hypothesis testing confirms Multinomial NB performs significantly better (p < 0.001) on this mixed SMS/Email dataset.

## 📁 Project Structure
- `data/` — Dataset loaders and stratified train/test splitting
- `preprocessing/` — NLTK-based text cleaning, tokenization, and pipeline
- `models/` — Raw implementations of Multinomial and Bernoulli Naïve Bayes
- `evaluation/` — Implementations of metrics, K-Fold CV, and hypothesis testing
- `visualization/` — Matplotlib/Seaborn implementations for word clouds and charts
- `outputs/` — Generated analysis plots
- `app.py` — Streamlit interactive UI
- `main.py` — Full pipeline entry point
