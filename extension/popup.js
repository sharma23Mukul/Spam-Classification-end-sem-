document.addEventListener('DOMContentLoaded', () => {
  const checkBtn = document.getElementById('checkBtn');
  const resultDiv = document.getElementById('result');
  const predBadge = document.getElementById('pred-badge');
  const metricsBox = document.getElementById('metricsbox');
  const loading = document.getElementById('loading');
  const errorMsg = document.getElementById('error');

  // Ensure FastAPI server URL is correct
  const API_URL = "https://spam-classification-end-sem.onrender.com/predict";

  checkBtn.addEventListener('click', async () => {
    // Reset UI
    resultDiv.style.display = 'none';
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
        throw new Error("Could not detect any email text. Please open an email in Gmail.");
      }

      // 3. Send text to local FastAPI endpoint
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: emailText })
      });

      if (!response.ok) {
        throw new Error(`API Error: Is the FastAPI server running on port 8000?`);
      }

      const data = await response.json();

      // 4. Update UI with results
      loading.style.display = 'none';
      resultDiv.style.display = 'block';

      // Strip previous classes
      resultDiv.className = '';

      if (data.prediction === 'spam') {
        resultDiv.classList.add('spam');
        predBadge.innerText = '🔴 SPAM DETECTED';
      } else if (data.prediction === 'ham') {
        resultDiv.classList.add('ham');
        predBadge.innerText = '🟢 SAFE (HAM)';
      } else {
        resultDiv.classList.add('uncertain');
        predBadge.innerText = '⚠️ UNCERTAIN';
      }

      let confPercent = (data.confidence * 100).toFixed(1);
      metricsBox.innerHTML = `
        <div style="margin-top: 5px;"><strong>Confidence:</strong> ${confPercent}%</div>
        <div style="margin-top: 3px; color: #666; font-style: italic;">
          P(Spam) = ${data.probabilities.spam.toFixed(4)}<br>
          P(Ham) = ${data.probabilities.ham.toFixed(4)}
        </div>
        <div style="margin-top: 5px; font-size: 11px;">
          ${data.explanation}
        </div>
      `;

    } catch (err) {
      loading.style.display = 'none';
      errorMsg.innerText = err.message;
      errorMsg.style.display = 'block';
    } finally {
      checkBtn.disabled = false;
    }
  });
});

// This function is executed INSIDE the Gmail webpage context
function extractEmailText() {
  // Gmail typically puts the email body inside elements with class 'a3s' or 'ii gt'
  // Or if it's an open thread, it tries to grab the visible text.
  const emailBodies = document.querySelectorAll('.a3s.aiL');

  if (emailBodies.length > 0) {
    // Get the last one (usually the most recent email in the thread)
    return emailBodies[emailBodies.length - 1].innerText;
  }

  // Fallback: grab all text on the page (messy but works as a fallback)
  return window.getSelection().toString() || document.body.innerText.substring(0, 5000);
}
