/**
 * Sentiment Analysis Studio - Frontend Application Logic
 * Integrates with FastAPI backend endpoints:
 *  - GET /api/health
 *  - POST /api/analyze
 *  - POST /api/analyze/batch
 */

// Determine API base URL: defaults to local FastAPI port 8000
const API_BASE_URL = window.location.origin.includes(':8000')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

const MIN_WORDS = 1;
const MAX_WORDS = 500;
const MAX_BATCH_SIZE = 10;

// Emoji and styling map for sentiments
const SENTIMENT_META = {
  positive: { emoji: '😊', label: 'Positive', cssClass: 'positive' },
  neutral:  { emoji: '😐', label: 'Neutral',  cssClass: 'neutral' },
  negative: { emoji: '😞', label: 'Negative', cssClass: 'negative' }
};

// Sample texts for quick demonstration
const SAMPLE_TEXTS = {
  positive: 'I absolutely love this new update! It runs exceptionally fast and makes my daily workflow so much easier.',
  neutral:  'The weekly engineering sync is scheduled for Thursday at 3:00 PM in meeting room B.',
  negative: 'The recent build is completely broken and crashes every time I try to import my dataset.'
};

const BATCH_DEFAULT_SAMPLES = [
  'The customer support team resolved my issue within minutes and was wonderfully helpful!',
  'The parcel arrived at the local sorting distribution center this morning.',
  'Terrible user experience. The app constantly freezes and loses all my unsaved progress.'
];

// -------------------------------------------------------------
// DOM Elements
// -------------------------------------------------------------
const healthIndicator = document.getElementById('healthIndicator');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const connectionAlert = document.getElementById('connectionAlert');

// Tabs
const singleTabBtn = document.getElementById('singleTabBtn');
const batchTabBtn = document.getElementById('batchTabBtn');
const singleTab = document.getElementById('singleTab');
const batchTab = document.getElementById('batchTab');

// Single Analysis
const singleForm = document.getElementById('singleForm');
const singleTextInput = document.getElementById('singleTextInput');
const singleWordCounter = document.getElementById('singleWordCounter');
const singleCharCounter = document.getElementById('singleCharCounter');
const singleSubmitBtn = document.getElementById('singleSubmitBtn');
const singleSpinner = document.getElementById('singleSpinner');
const singleClearBtn = document.getElementById('singleClearBtn');
const singleErrorBox = document.getElementById('singleErrorBox');
const singleErrorMessage = document.getElementById('singleErrorMessage');

const singleResultCard = document.getElementById('singleResultCard');
const sentimentBadge = document.getElementById('sentimentBadge');
const sentimentEmoji = document.getElementById('sentimentEmoji');
const sentimentName = document.getElementById('sentimentName');
const confidenceValue = document.getElementById('confidenceValue');
const wordCountValue = document.getElementById('wordCountValue');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceBarPercent = document.getElementById('confidenceBarPercent');
const analyzedTextPreview = document.getElementById('analyzedTextPreview');
const singleResultTime = document.getElementById('singleResultTime');

// Batch Analysis
const batchForm = document.getElementById('batchForm');
const batchItemsContainer = document.getElementById('batchItemsContainer');
const addBatchItemBtn = document.getElementById('addBatchItemBtn');
const batchCountInfo = document.getElementById('batchCountInfo');
const loadBatchSampleBtn = document.getElementById('loadBatchSampleBtn');
const batchSubmitBtn = document.getElementById('batchSubmitBtn');
const batchSpinner = document.getElementById('batchSpinner');
const clearBatchBtn = document.getElementById('clearBatchBtn');
const batchErrorBox = document.getElementById('batchErrorBox');
const batchErrorMessage = document.getElementById('batchErrorMessage');

const batchResultCard = document.getElementById('batchResultCard');
const batchTotalCount = document.getElementById('batchTotalCount');
const batchSummaryStats = document.getElementById('batchSummaryStats');
const batchResultTime = document.getElementById('batchResultTime');
const countPos = document.getElementById('countPos');
const countNeu = document.getElementById('countNeu');
const countNeg = document.getElementById('countNeg');
const batchResultsList = document.getElementById('batchResultsList');

// -------------------------------------------------------------
// Helper: Count whitespace-separated words
// -------------------------------------------------------------
function countWords(text) {
  if (!text || !text.trim()) return 0;
  return text.trim().split(/\s+/).length;
}

// -------------------------------------------------------------
// Health Check Polling
// -------------------------------------------------------------
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    if (data.model_loaded) {
      statusDot.className = 'status-dot status-online';
      const shortModel = data.model_name.includes('/') ? data.model_name.split('/')[1] : data.model_name;
      statusText.textContent = `Online (${shortModel} • ${data.device.toUpperCase()})`;
      connectionAlert.classList.add('hidden');
    } else {
      statusDot.className = 'status-dot status-loading';
      statusText.textContent = 'Model Initializing...';
    }
  } catch (err) {
    statusDot.className = 'status-dot status-offline';
    statusText.textContent = 'API Offline';
    connectionAlert.classList.remove('hidden');
  }
}

// Check health immediately and every 10 seconds
checkHealth();
setInterval(checkHealth, 10000);

// -------------------------------------------------------------
// Tab Switching
// -------------------------------------------------------------
function switchTab(target) {
  if (target === 'single') {
    singleTabBtn.classList.add('active');
    singleTabBtn.setAttribute('aria-selected', 'true');
    batchTabBtn.classList.remove('active');
    batchTabBtn.setAttribute('aria-selected', 'false');

    singleTab.classList.add('active');
    batchTab.classList.remove('active');
  } else {
    batchTabBtn.classList.add('active');
    batchTabBtn.setAttribute('aria-selected', 'true');
    singleTabBtn.classList.remove('active');
    singleTabBtn.setAttribute('aria-selected', 'false');

    batchTab.classList.add('active');
    singleTab.classList.remove('active');
  }
}

singleTabBtn.addEventListener('click', () => switchTab('single'));
batchTabBtn.addEventListener('click', () => switchTab('batch'));

// -------------------------------------------------------------
// Single Analysis Handlers
// -------------------------------------------------------------
function updateSingleCounters() {
  const text = singleTextInput.value;
  const words = countWords(text);
  const chars = text.length;

  singleWordCounter.textContent = `${words} word${words === 1 ? '' : 's'} (${MIN_WORDS} - ${MAX_WORDS} allowed)`;
  singleCharCounter.textContent = `${chars} character${chars === 1 ? '' : 's'}`;

  if (words > MAX_WORDS) {
    singleWordCounter.classList.add('limit-exceeded');
  } else {
    singleWordCounter.classList.remove('limit-exceeded');
  }
}

singleTextInput.addEventListener('input', updateSingleCounters);

// Quick sample chips
document.querySelectorAll('.sample-chips .chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const sampleKey = chip.dataset.sample;
    if (SAMPLE_TEXTS[sampleKey]) {
      singleTextInput.value = SAMPLE_TEXTS[sampleKey];
      updateSingleCounters();
      singleTextInput.focus();
    }
  });
});

// Clear single form
singleClearBtn.addEventListener('click', () => {
  singleTextInput.value = '';
  updateSingleCounters();
  singleResultCard.classList.add('hidden');
  singleErrorBox.classList.add('hidden');
  singleTextInput.focus();
});

// Single Submit
singleForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  singleErrorBox.classList.add('hidden');
  singleResultCard.classList.add('hidden');

  const text = singleTextInput.value.trim();
  const words = countWords(text);

  // Client-side validation check
  if (words < MIN_WORDS) {
    showSingleError('Please enter at least 1 word.');
    return;
  }
  if (words > MAX_WORDS) {
    showSingleError(`Input has ${words} words. The maximum allowed limit is ${MAX_WORDS} words.`);
    return;
  }

  // Set loading state
  setSingleLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 422 && data.details) {
        const issues = data.details.map(d => d.message).join(' | ');
        throw new Error(`Validation Error: ${issues}`);
      } else {
        throw new Error(data.detail || data.error || `HTTP error ${response.status}`);
      }
    }

    displaySingleResult(data);
  } catch (err) {
    showSingleError(err.message || 'Failed to communicate with Sentiment Analysis API.');
  } finally {
    setSingleLoading(false);
  }
});

function setSingleLoading(isLoading) {
  if (isLoading) {
    singleSubmitBtn.disabled = true;
    singleSpinner.classList.remove('hidden');
  } else {
    singleSubmitBtn.disabled = false;
    singleSpinner.classList.add('hidden');
  }
}

function showSingleError(msg) {
  singleErrorMessage.textContent = msg;
  singleErrorBox.classList.remove('hidden');
}

function displaySingleResult(data) {
  const sentimentKey = data.sentiment.toLowerCase();
  const meta = SENTIMENT_META[sentimentKey] || { emoji: '🔍', label: data.sentiment, cssClass: 'neutral' };
  const percentStr = `${(data.confidence * 100).toFixed(1)}%`;

  // Badge
  sentimentBadge.className = `sentiment-badge-large ${meta.cssClass}`;
  sentimentEmoji.textContent = meta.emoji;
  sentimentName.textContent = meta.label;

  // Metrics
  confidenceValue.textContent = percentStr;
  wordCountValue.textContent = data.word_count;

  // Progress Bar
  confidenceBarPercent.textContent = percentStr;
  confidenceFill.className = `confidence-fill ${meta.cssClass}`;
  confidenceFill.style.width = '0%';
  setTimeout(() => {
    confidenceFill.style.width = percentStr;
  }, 50);

  // Original text & time
  analyzedTextPreview.textContent = `"${data.text}"`;
  singleResultTime.textContent = `Analyzed at ${new Date().toLocaleTimeString()}`;

  singleResultCard.classList.remove('hidden');
  singleResultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// -------------------------------------------------------------
// Batch Analysis Handlers
// -------------------------------------------------------------
function renderBatchItem(initialText = '') {
  const currentCount = batchItemsContainer.querySelectorAll('.batch-row').length;
  if (currentCount >= MAX_BATCH_SIZE) return;

  const row = document.createElement('div');
  row.className = 'batch-row';

  const indexPill = document.createElement('span');
  indexPill.className = 'batch-index-pill';
  indexPill.textContent = currentCount + 1;

  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = `Text #${currentCount + 1} to analyze (1 - 500 words)...`;
  input.value = initialText;
  input.required = true;

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'batch-btn-remove';
  removeBtn.innerHTML = '&times;';
  removeBtn.title = 'Remove item';
  removeBtn.addEventListener('click', () => {
    row.remove();
    updateBatchIndices();
  });

  row.appendChild(indexPill);
  row.appendChild(input);
  row.appendChild(removeBtn);

  batchItemsContainer.appendChild(row);
  updateBatchIndices();
}

function updateBatchIndices() {
  const rows = batchItemsContainer.querySelectorAll('.batch-row');
  rows.forEach((r, idx) => {
    const pill = r.querySelector('.batch-index-pill');
    const input = r.querySelector('input');
    if (pill) pill.textContent = idx + 1;
    if (input && !input.value) {
      input.placeholder = `Text #${idx + 1} to analyze (1 - 500 words)...`;
    }
  });

  const count = rows.length;
  batchCountInfo.textContent = `${count} of ${MAX_BATCH_SIZE} texts`;
  addBatchItemBtn.disabled = count >= MAX_BATCH_SIZE;
}

addBatchItemBtn.addEventListener('click', () => renderBatchItem());

// Populate 3 samples on button click
loadBatchSampleBtn.addEventListener('click', () => {
  batchItemsContainer.innerHTML = '';
  BATCH_DEFAULT_SAMPLES.forEach(sample => renderBatchItem(sample));
});

// Clear batch
clearBatchBtn.addEventListener('click', () => {
  batchItemsContainer.innerHTML = '';
  renderBatchItem();
  batchResultCard.classList.add('hidden');
  batchErrorBox.classList.add('hidden');
});

// Batch Submit
batchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  batchErrorBox.classList.add('hidden');
  batchResultCard.classList.add('hidden');

  const rows = batchItemsContainer.querySelectorAll('.batch-row input');
  const texts = [];

  for (let i = 0; i < rows.length; i++) {
    const val = rows[i].value.trim();
    if (!val) {
      showBatchError(`Text #${i + 1} is empty. Please enter text or remove the row.`);
      rows[i].focus();
      return;
    }
    const words = countWords(val);
    if (words > MAX_WORDS) {
      showBatchError(`Text #${i + 1} has ${words} words, which exceeds the limit of ${MAX_WORDS} words.`);
      rows[i].focus();
      return;
    }
    texts.push(val);
  }

  if (texts.length === 0) {
    showBatchError('Please add at least 1 text item.');
    return;
  }

  if (texts.length > MAX_BATCH_SIZE) {
    showBatchError(`Batch cannot exceed ${MAX_BATCH_SIZE} texts.`);
    return;
  }

  setBatchLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 422 && data.details) {
        const issues = data.details.map(d => d.message).join(' | ');
        throw new Error(`Validation Error: ${issues}`);
      } else {
        throw new Error(data.detail || data.error || `HTTP error ${response.status}`);
      }
    }

    displayBatchResults(data);
  } catch (err) {
    showBatchError(err.message || 'Failed to process batch sentiment analysis.');
  } finally {
    setBatchLoading(false);
  }
});

function setBatchLoading(isLoading) {
  if (isLoading) {
    batchSubmitBtn.disabled = true;
    batchSpinner.classList.remove('hidden');
  } else {
    batchSubmitBtn.disabled = false;
    batchSpinner.classList.add('hidden');
  }
}

function showBatchError(msg) {
  batchErrorMessage.textContent = msg;
  batchErrorBox.classList.remove('hidden');
}

function displayBatchResults(data) {
  batchTotalCount.textContent = data.total;
  batchResultTime.textContent = `Completed at ${new Date().toLocaleTimeString()}`;

  // Calculate sentiment distributions
  let pos = 0;
  let neu = 0;
  let neg = 0;

  batchResultsList.innerHTML = '';

  data.results.forEach((item, idx) => {
    const sentimentKey = item.sentiment.toLowerCase();
    if (sentimentKey === 'positive') pos++;
    else if (sentimentKey === 'neutral') neu++;
    else if (sentimentKey === 'negative') neg++;

    const meta = SENTIMENT_META[sentimentKey] || { emoji: '🔍', label: item.sentiment, cssClass: 'neutral' };
    const confPercent = `${(item.confidence * 100).toFixed(1)}%`;

    const row = document.createElement('div');
    row.className = 'batch-result-item';
    row.innerHTML = `
      <div class="batch-result-left">
        <span class="batch-index-pill">${idx + 1}</span>
        <span class="batch-result-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</span>
      </div>
      <div class="batch-result-right">
        <span class="badge-sm ${meta.cssClass}">${meta.emoji} ${meta.label}</span>
        <div class="batch-result-meta">
          <strong>${confPercent}</strong>
          <div>${item.word_count} word${item.word_count === 1 ? '' : 's'}</div>
        </div>
      </div>
    `;

    batchResultsList.appendChild(row);
  });

  countPos.textContent = pos;
  countNeu.textContent = neu;
  countNeg.textContent = neg;

  batchResultCard.classList.remove('hidden');
  batchResultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (m) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  }[m]));
}

// -------------------------------------------------------------
// Initialize Default Batch Inputs
// -------------------------------------------------------------
// Pre-populate with 3 items by default for convenience
BATCH_DEFAULT_SAMPLES.forEach(sample => renderBatchItem(sample));
