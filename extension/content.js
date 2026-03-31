// Function to extract text and add badge to a row
function classifyAndBadgeRow(row) {
  // Prevent duplicate checks
  if (row.dataset.spamChecked === "true") return;
  row.dataset.spamChecked = "true";

  // Gmail DOM structure:
  // Subject is usually inside an element with class .bog
  // Snippet is usually inside an element with class .y2
  const subjectWrapper = row.querySelector('.bog');
  const snippetWrapper = row.querySelector('.y2');
  
  if (!subjectWrapper) return;
  
  const subjectText = subjectWrapper.innerText || "";
  const snippetText = snippetWrapper ? (snippetWrapper.innerText || "") : "";
  const fullText = subjectText + " " + snippetText;
  
  if (fullText.trim().length === 0) return;
  
  // Send message to background script to avoid CORS
  chrome.runtime.sendMessage({ action: "classifyEmail", text: fullText }, (response) => {
    // Suppress unhandled runtime error if extension context invalidates
    if (chrome.runtime.lastError) return;
    
    if (response && response.success) {
      const pred = response.data.prediction;
      // Get confidence and format it
      const conf = response.data.confidence ? (response.data.confidence * 100).toFixed(0) : "0";
      
      const badge = document.createElement('span');
      badge.className = `ai-spam-badge badge-${pred}`;
      badge.innerText = pred === 'spam' ? `[SPAM ${conf}%]` : `[HAM]`;
      
      // Inject the badge into the Gmail DOM, directly before the subject
      subjectWrapper.insertBefore(badge, subjectWrapper.firstChild);
      
      if (pred === 'spam') {
        row.style.backgroundColor = 'rgba(255, 0, 0, 0.05)';
      }
    }
  });
}

// Observe Gmail's Inbox DOM changes to detect newly added emails (scrolling, new mail, etc.)
const observer = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    mutation.addedNodes.forEach((node) => {
      // Check if node is an element
      if (node.nodeType === Node.ELEMENT_NODE) {
        // If the added node itself is an email row
        if (node.matches && node.matches('tr.zA')) {
          classifyAndBadgeRow(node);
        } else if (node.querySelectorAll) {
          // Check children for email rows
          const rows = node.querySelectorAll('tr.zA');
          rows.forEach(row => classifyAndBadgeRow(row));
        }
      }
    });
  });
});

// Start observing the body for changes
observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Classify already existing rows on initial page load
setTimeout(() => {
  const existingRows = document.querySelectorAll('tr.zA');
  existingRows.forEach(row => classifyAndBadgeRow(row));
}, 3000); // 3-second delay to allow Gmail to render initial rows

console.log("Spam Classifier: Background scanner active.");
