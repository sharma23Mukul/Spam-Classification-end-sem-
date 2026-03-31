const API_URL = "https://spam-classification-end-sem.onrender.com/predict";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classifyEmail") {
    fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: request.text })
    })
    .then(response => {
      if (!response.ok) throw new Error("API error");
      return response.json();
    })
    .then(data => sendResponse({ success: true, data: data }))
    .catch(error => sendResponse({ success: false, error: error.toString() }));
    
    // Return true to indicate we wish to send a response asynchronously
    return true;
  }
});
