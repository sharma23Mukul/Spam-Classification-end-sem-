document.addEventListener('DOMContentLoaded', async () => {
  const checkBtn = document.getElementById('checkBtn');
  const resultCard = document.getElementById('result-card');
  const predBadge = document.getElementById('pred-badge');
  const confVal = document.getElementById('conf-val');
  const spamProb = document.getElementById('spam-prob');
  const explanationText = document.getElementById('explanation-text');
  const loading = document.getElementById('loading');
  const errorMsg = document.getElementById('error');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdVal = document.getElementById('threshold-val');
  const secretToggle = document.getElementById('secret-toggle');
  
  let useSenderBoost = false;

  // Ensure FastAPI server URL is correct
  const API_URL = "https://spam-classification-end-sem.onrender.com/predict";

  // Load saved settings
  const settings = await chrome.storage.local.get(['hamThreshold', 'useSenderBoost']);
  if (settings.hamThreshold) {
    thresholdSlider.value = settings.hamThreshold;
    thresholdVal.innerText = settings.hamThreshold;
  }
  if (settings.useSenderBoost) {
    useSenderBoost = settings.useSenderBoost;
    if (useSenderBoost) {
        secretToggle.classList.add('active');
    }
  }

  // Secret Toggle Logic
  secretToggle.addEventListener('click', () => {
      useSenderBoost = !useSenderBoost;
      if (useSenderBoost) {
          secretToggle.classList.add('active');
      } else {
          secretToggle.classList.remove('active');
      }
      chrome.storage.local.set({ useSenderBoost: useSenderBoost });
  });

  // Update threshold UI and save
  thresholdSlider.addEventListener('input', (e) => {
    thresholdVal.innerText = e.target.value;
    chrome.storage.local.set({ hamThreshold: parseFloat(e.target.value) });
  });

  checkBtn.addEventListener('click', async () => {
    // Reset UI
    resultCard.style.display = 'none';
    errorMsg.style.display = 'none';
    loading.style.display = 'block';
    checkBtn.disabled = true;

    try {
      // 1. Get the active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      // 2. Inject a script to pull the text from Gmail
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractEmailText
      });

      const emailText = results[0]?.result;

      if (!emailText || emailText.trim().length === 0) {
        throw new Error("No email text detected. Open a Gmail message first.");
      }

      // 3. Send text to local FastAPI endpoint
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          message: emailText,
          decision_threshold: parseFloat(thresholdSlider.value),
          use_sender_boost: useSenderBoost
        })
      });

      let data;
      const responseText = await response.text();
      
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        console.error("Failed to parse JSON response:", responseText);
        throw new Error(`Invalid API Response (Code ${response.status}). The server might be starting up or experiencing an issue.`);
      }

      if (!response.ok) {
        throw new Error(data.detail || `API Error: ${response.status}`);
      }

      // 4. Update UI with results
      loading.style.display = 'none';
      resultCard.style.display = 'block';

      // Clear previous classes
      predBadge.className = 'badge';

      if (data.prediction === 'spam') {
        predBadge.classList.add('spam');
        predBadge.innerText = '🔴 Spam Detected';
      } else if (data.prediction === 'ham') {
        predBadge.classList.add('ham');
        predBadge.innerText = '🟢 Legitimate';
      } else {
        predBadge.classList.add('uncertain');
        predBadge.innerText = '⚠️ Uncertain';
      }

      confVal.innerText = (data.confidence * 100).toFixed(1) + '%';
      spamProb.innerText = data.probabilities.spam.toFixed(3);
      explanationText.innerText = data.explanation;

    } catch (err) {
      loading.style.display = 'none';
      errorMsg.innerText = "Error: " + err.message;
      errorMsg.style.display = 'block';
    } finally {
      checkBtn.disabled = false;
    }
  });
});

// This function is executed INSIDE the Gmail webpage context
function extractEmailText() {
  const emailBodies = document.querySelectorAll('.a3s.aiL');
  if (emailBodies.length > 0) {
    return emailBodies[emailBodies.length - 1].innerText;
  }
  return window.getSelection().toString() || document.body.innerText.substring(0, 5000);
}
