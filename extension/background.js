const API_URL = "https://spam-classification-end-sem.onrender.com/predict";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classifyEmail") {
    // Load user settings before making the API call
    chrome.storage.local.get(['hamThreshold', 'useSenderBoost'], (settings) => {
      const threshold = settings.hamThreshold || 0.65;
      const senderBoost = settings.useSenderBoost || false;

      fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: request.text,
          decision_threshold: threshold,
          use_sender_boost: senderBoost
        })
      })
      .then(response => {
        if (!response.ok) throw new Error("API error");
        return response.json();
      })
      .then(data => sendResponse({ success: true, data: data }))
      .catch(error => sendResponse({ success: false, error: error.toString() }));
    });

    // Return true to indicate we wish to send a response asynchronously
    return true;
  }
});
