# Project Structure

```text
Probability Project/
├── README.md
├── app.py (Streamlit UI)
├── api.py (FastAPI Server)
├── main.py (Training & Eval Pipeline)
├── requirements.txt
├── models/
│   ├── base.py
│   ├── multinomial_nb.py
│   └── bernoulli_nb.py
├── preprocessing/
│   └── pipeline.py
├── data/
│   └── loader.py
├── evaluation/
│   ├── metrics.py
│   ├── cross_validation.py
│   └── hypothesis_testing.py
└── extension/
    ├── manifest.json
    ├── background.js
    ├── content.js
    └── popup.js
```

# Core Model Implementation

## `models/base.py`
```python
import math
from abc import ABC, abstractmethod

class NaiveBayesBase(ABC):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = ['ham', 'spam']
        self.log_priors = {}
        self.vocabulary = []
        self._is_fitted = False

    def fit(self, processed_data, vocabulary):
        self.vocabulary = vocabulary
        self.vocab_set = set(vocabulary)

        class_counts = {cls: 0 for cls in self.classes}
        for label, _ in processed_data:
            class_counts[label] += 1

        total = len(processed_data)
        for cls in self.classes:
            self.log_priors[cls] = math.log(class_counts[cls] / total)

        self._compute_likelihoods(processed_data)
        self._is_fitted = True

    @abstractmethod
    def _compute_likelihoods(self, processed_data):
        pass

    @abstractmethod
    def _compute_log_likelihood(self, tokens, cls):
        pass

    def predict(self, tokens):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        log_posteriors = {}
        for cls in self.classes:
            log_posteriors[cls] = self.log_priors[cls] + self._compute_log_likelihood(tokens, cls)

        max_log = max(log_posteriors.values())
        log_sum = max_log + math.log(sum(math.exp(lp - max_log) for lp in log_posteriors.values()))

        probabilities = {}
        for cls in self.classes:
            probabilities[cls] = math.exp(log_posteriors[cls] - log_sum)

        predicted = max(probabilities, key=probabilities.get)
        return predicted, probabilities

    def predict_with_confidence(self, tokens, confidence_threshold=0.70):
        pred, probs = self.predict(tokens)
        confidence = probs[pred]
        is_confident = confidence >= confidence_threshold
        
        final_pred = pred if is_confident else 'uncertain'
        
        return {
            'prediction': final_pred,
            'confidence': confidence,
            'probabilities': probs,
            'is_confident': is_confident,
            'explanation': f"Confident {pred}" if is_confident else "UNCERTAIN"
        }
```

## `models/multinomial_nb.py`
```python
import math
from collections import Counter
from models.base import NaiveBayesBase

class MultinomialNaiveBayes(NaiveBayesBase):
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        self.log_likelihoods = {'spam': {}, 'ham': {}}
        self.class_word_counts = {}

    def _compute_likelihoods(self, processed_data):
        vocab_size = len(self.vocabulary)
        word_counts = {'spam': Counter(), 'ham': Counter()}
        
        for label, tokens in processed_data:
            word_counts[label].update(tokens)

        for cls in self.classes:
            self.class_word_counts[cls] = sum(word_counts[cls].values())
            n_class = self.class_word_counts[cls]
            denominator = n_class + self.alpha * vocab_size

            for word in self.vocabulary:
                numerator = word_counts[cls].get(word, 0) + self.alpha
                self.log_likelihoods[cls][word] = math.log(numerator / denominator)

    def _compute_log_likelihood(self, tokens, cls):
        log_likelihood = 0.0
        vocab_size = len(self.vocabulary)
        n_class = self.class_word_counts[cls]

        for token in tokens:
            if token in self.log_likelihoods[cls]:
                log_likelihood += self.log_likelihoods[cls][token]
            else:
                log_likelihood += math.log(self.alpha / (n_class + self.alpha * vocab_size))

        return log_likelihood
```

## `models/bernoulli_nb.py`
```python
import math
from collections import Counter
from models.base import NaiveBayesBase

class BernoulliNaiveBayes(NaiveBayesBase):
    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)
        self.log_prob_present = {'spam': {}, 'ham': {}}
        self.log_prob_absent = {'spam': {}, 'ham': {}}
        self.class_doc_counts = {}

    def _compute_likelihoods(self, processed_data):
        self.class_doc_counts = {'spam': 0, 'ham': 0}
        doc_freq = {'spam': Counter(), 'ham': Counter()}
        
        for label, tokens in processed_data:
            self.class_doc_counts[label] += 1
            doc_freq[label].update(set(tokens))

        self._absent_baseline = {}

        for cls in self.classes:
            n_docs = self.class_doc_counts[cls]
            baseline = 0.0

            for word in self.vocabulary:
                numerator = doc_freq[cls].get(word, 0) + self.alpha
                denominator = n_docs + 2 * self.alpha

                prob_present = numerator / denominator
                prob_absent = 1.0 - prob_present

                log_p = math.log(prob_present)
                log_1mp = math.log(prob_absent)

                self.log_prob_present[cls][word] = log_p
                self.log_prob_absent[cls][word] = log_1mp
                baseline += log_1mp

            self._absent_baseline[cls] = baseline

    def _compute_log_likelihood(self, tokens, cls):
        log_likelihood = self._absent_baseline[cls]
        token_set = set(tokens) & self.vocab_set

        log_present = self.log_prob_present[cls]
        log_absent = self.log_prob_absent[cls]

        for word in token_set:
            log_likelihood += log_present[word] - log_absent[word]

        return log_likelihood
```

# Preprocessing Pipeline

## `preprocessing/pipeline.py`
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

def preprocess_corpus(data):
    return [(label, preprocess(msg)) for label, msg in data]

def build_vocabulary(processed_data):
    vocab = set()
    for _, tokens in processed_data:
        vocab.update(tokens)
    return sorted(vocab)
```

# API & Backend

## `api.py`
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.pipeline import preprocess

app = FastAPI(title="Probabilistic Spam Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_model.pkl")
loaded_model = None

@app.on_event("startup")
def load_saved_model():
    global loaded_model
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
        loaded_model = bundle["model"]

class PredictionRequest(BaseModel):
    message: str

@app.post("/predict")
def predict_spam(request: PredictionRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    tokens = preprocess(request.message)
    if not tokens:
        return {"prediction": "uncertain", "confidence": 0.0, "is_confident": False, "probabilities": {"spam": 0.5, "ham": 0.5}}

    result = loaded_model.predict_with_confidence(tokens, confidence_threshold=0.70)
    return result
```

# Extension Example

## `extension/content.js`
```javascript
function classifyAndBadgeRow(row) {
  if (row.dataset.spamChecked === "true") return;
  row.dataset.spamChecked = "true";

  const subjectWrapper = row.querySelector('.bog');
  const snippetWrapper = row.querySelector('.y2');
  if (!subjectWrapper) return;
  
  const fullText = (subjectWrapper.innerText || "") + " " + (snippetWrapper ? snippetWrapper.innerText || "" : "");
  if (fullText.trim().length === 0) return;
  
  chrome.runtime.sendMessage({ action: "classifyEmail", text: fullText }, (response) => {
    if (chrome.runtime.lastError) return;
    if (response && response.success) {
      const pred = response.data.prediction;
      const conf = response.data.confidence ? (response.data.confidence * 100).toFixed(0) : "0";
      
      const badge = document.createElement('span');
      badge.className = `ai-spam-badge badge-${pred}`;
      badge.innerText = pred === 'spam' ? `[SPAM ${conf}%]` : `[HAM]`;
      
      subjectWrapper.insertBefore(badge, subjectWrapper.firstChild);
      if (pred === 'spam') row.style.backgroundColor = 'rgba(255, 0, 0, 0.05)';
    }
  });
}

const observer = new MutationObserver((mutations) => {
  mutations.forEach((m) => m.addedNodes.forEach((node) => {
    if (node.nodeType === Node.ELEMENT_NODE) {
      if (node.matches && node.matches('tr.zA')) classifyAndBadgeRow(node);
      else if (node.querySelectorAll) node.querySelectorAll('tr.zA').forEach(classifyAndBadgeRow);
    }
  }));
});
observer.observe(document.body, { childList: true, subtree: true });
```
