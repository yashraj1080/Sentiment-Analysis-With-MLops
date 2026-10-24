/**
 * Sentiment Analysis MLOps Pipeline — Interactive Dashboard Engine
 * Handles real-time inference, batch analysis, live metrics polling,
 * and statistical drift simulation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Preset reviews
  const PRESETS = {
    masterpiece: "This movie is an absolute masterpiece! Brilliant acting, magnificent visual direction, and an unforgettable story.",
    disaster: "Terrible movie. Boring plot, dreadful acting, cheap effects, and a complete waste of my time and money.",
    product: "Fantastic product! Sleek design, outstanding build quality, high performance, and worth every dollar.",
    support: "Worst customer service ever. Kept me on hold for an hour, rude representatives, and refused my refund.",
    mixed: "The cinematography was stunning and the soundtrack was beautiful, but the pacing was sluggish and the plot was full of holes.",
  };

  // State
  let isPolling = true;
  let pollTimer = null;
  let batchStore = [];

  // DOM Elements - Tabs
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  function activateTab(targetId) {
    tabBtns.forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === targetId));
    tabPanes.forEach(p => p.classList.toggle('active', p.id === targetId));
    const activeButton = [...tabBtns].find(b => b.getAttribute('data-tab') === targetId);
    const pageLabel = document.getElementById('pageLabel');
    if (activeButton && pageLabel) pageLabel.textContent = activeButton.querySelector('span:nth-child(2)').textContent;
  }

  // DOM Elements - Playground
  const textInput = document.getElementById('sentimentText');
  const charCount = document.getElementById('charCount');
  const btnPredict = document.getElementById('btnPredict');
  const btnClear = document.getElementById('btnClear');
  const chips = document.querySelectorAll('.chip');
  const resultEmpty = document.getElementById('resultEmpty');
  const resultActive = document.getElementById('resultActive');
  const sentimentBadge = document.getElementById('sentimentBadge');
  const sentimentIcon = document.getElementById('sentimentIcon');
  const sentimentTextLabel = document.getElementById('sentimentTextLabel');
  const gaugeProgress = document.getElementById('gaugeProgress');
  const gaugePercent = document.getElementById('gaugePercent');
  const probPosFill = document.getElementById('probPosFill');
  const probPosText = document.getElementById('probPosText');
  const probNegFill = document.getElementById('probNegFill');
  const probNegText = document.getElementById('probNegText');
  const latencyBadge = document.getElementById('latencyBadge');
  const emotionStatus = document.getElementById('emotionStatus');
  const emotionResults = document.getElementById('emotionResults');
  const emotionSegments = document.getElementById('emotionSegments');
  const modelVersionBadge = document.getElementById('modelVersionBadge');

  // DOM Elements - Batch
  const batchInput = document.getElementById('batchInput');
  const btnRunBatch = document.getElementById('btnRunBatch');
  const batchTableBody = document.getElementById('batchTableBody');
  const batchEmpty = document.getElementById('batchEmpty');
  const btnExportBatch = document.getElementById('btnExportBatch');
  const batchFilter = document.getElementById('batchFilter');

  // DOM Elements - Telemetry Deck
  const statTotalReq = document.getElementById('statTotalReq');
  const statTotalPred = document.getElementById('statTotalPred');
  const statPosNegRatio = document.getElementById('statPosNegRatio');
  const statAvgLatency = document.getElementById('statAvgLatency');
  const statUptime = document.getElementById('statUptime');
  const splitPositive = document.getElementById('splitPositive');
  const splitNegative = document.getElementById('splitNegative');
  const legendPos = document.getElementById('legendPos');
  const legendNeg = document.getElementById('legendNeg');
  const togglePolling = document.getElementById('togglePolling');

  // DOM Elements - Drift
  const btnSimulateDrift = document.getElementById('btnSimulateDrift');
  const driftConsole = document.getElementById('driftConsole');

  // -------------------------------------------------------------------------
  // 1. Tab Navigation
  // -------------------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      activateTab(btn.getAttribute('data-tab'));
    });
  });
  document.querySelectorAll('[data-tab]:not(.tab-btn)').forEach(link => {
    link.addEventListener('click', () => activateTab(link.getAttribute('data-tab')));
  });

  // -------------------------------------------------------------------------
  // 2. Playground Logic
  // -------------------------------------------------------------------------
  // Character count update
  textInput.addEventListener('input', () => {
    charCount.textContent = `${textInput.value.length} / 5000`;
  });

  // Preset Chips Click
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.getAttribute('data-preset');
      if (PRESETS[presetKey]) {
        textInput.value = PRESETS[presetKey];
        charCount.textContent = `${textInput.value.length} / 5000`;
        textInput.focus();
      }
    });
  });

  // Clear button
  btnClear.addEventListener('click', () => {
    textInput.value = '';
    charCount.textContent = '0 / 5000';
    resultActive.style.display = 'none';
    resultEmpty.style.display = 'flex';
  });

  // Single Prediction API Call
  btnPredict.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
      alert('Please enter a review text or select one of the preset chips above.');
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = `<span class="spinner">⏳</span> Analyzing Sentiment...`;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Server returned status: ${response.status}`);
      }

      const data = await response.json();
      renderSingleResult(data);
      updateEmotionRead(text);
      refreshMetrics(); // Refresh telemetry stats immediately
    } catch (err) {
      console.error('Inference error:', err);
      showNotice(`Analysis unavailable: ${err.message}`);
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<span>⚡</span> Classify Sentiment`;
    }
  });

  function renderSingleResult(data) {
    resultEmpty.style.display = 'none';
    resultActive.style.display = 'flex';

    const isPos = data.sentiment.toLowerCase() === 'positive';
    sentimentBadge.className = `sentiment-badge-large ${isPos ? 'positive' : 'negative'}`;
    sentimentIcon.textContent = isPos ? '✨' : '⚠️';
    sentimentTextLabel.textContent = isPos ? 'POSITIVE SENTIMENT' : 'NEGATIVE SENTIMENT';

    // Circular gauge animation (perimeter of r=40 is 2 * PI * 40 ≈ 251.2)
    const conf = Math.max(0, Math.min(1, data.confidence));
    const percentVal = Math.round(conf * 100);
    const offset = 251.2 * (1 - conf);
    gaugeProgress.style.strokeDashoffset = offset;
    gaugeProgress.style.stroke = isPos ? 'var(--positive)' : 'var(--negative)';
    gaugePercent.textContent = `${percentVal}%`;

    // Probability breakdown bars
    const posPct = Math.round((data.probabilities.positive || 0) * 100);
    const negPct = Math.round((data.probabilities.negative || 0) * 100);

    probPosFill.style.width = `${posPct}%`;
    probPosText.textContent = `${posPct}%`;

    probNegFill.style.width = `${negPct}%`;
    probNegText.textContent = `${negPct}%`;

    // Latency & Model Version
    latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
    modelVersionBadge.textContent = `v${data.model_version}`;
  }

  async function updateEmotionRead(text) {
    emotionStatus.textContent = 'Reading emotional tone…';
    emotionResults.replaceChildren();
    emotionSegments.replaceChildren();
    try {
      const response = await fetch('/predict/emotions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Emotion model unavailable (${response.status})`);
      const detected = data.emotions.map(item => item.emotion.replace(/_/g, ' '));
      emotionStatus.textContent = data.emotions.length
        ? `Overall threshold-selected: ${detected.join(', ')}. Clause-level signals below can reveal shifts.`
        : data.is_neutral ? 'The statement reads as neutral.' : 'No emotion passed its tuned threshold; showing the strongest signals.';
      data.top_emotions.forEach(item => {
        const row = document.createElement('div');
        row.className = 'emotion-score-row';
        const heading = document.createElement('div');
        heading.className = 'emotion-score-heading';
        const name = document.createElement('span');
        name.textContent = item.emotion.replace(/_/g, ' ');
        const score = document.createElement('strong');
        score.textContent = `${Math.round(item.confidence * 100)}%`;
        heading.append(name, score);
        const track = document.createElement('div');
        track.className = 'emotion-score-track';
        const fill = document.createElement('div');
        fill.className = 'emotion-score-fill';
        fill.style.width = `${Math.round(item.confidence * 100)}%`;
        track.append(fill);
        row.append(heading, track);
        emotionResults.append(row);
      });
      data.segments.forEach(segment => {
        const card = document.createElement('article');
        card.className = 'emotion-segment-card';
        const quote = document.createElement('p');
        quote.textContent = segment.text;
        const labels = document.createElement('div');
        labels.className = 'emotion-segment-labels';
        segment.emotions.forEach(item => {
          const badge = document.createElement('span');
          badge.textContent = `${item.emotion.replace(/_/g, ' ')} · ${Math.round(item.confidence * 100)}%`;
          labels.append(badge);
        });
        card.append(quote, labels);
        emotionSegments.append(card);
      });
    } catch (error) {
      emotionStatus.textContent = error.message;
    }
  }

  // -------------------------------------------------------------------------
  // 3. Batch Analysis Lab
  // -------------------------------------------------------------------------
  btnRunBatch.addEventListener('click', async () => {
    const rawLines = batchInput.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (rawLines.length === 0) {
      alert('Please enter at least one review text in the batch textarea (one per line).');
      return;
    }

    btnRunBatch.disabled = true;
    btnRunBatch.innerHTML = `<span>⏳</span> Processing ${rawLines.length} Reviews...`;

    try {
      const response = await fetch('/predict/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: rawLines.slice(0, 100) }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Batch request failed: ${response.status}`);
      }

      const data = await response.json();
      batchStore = data.results;
      renderBatchTable(batchStore);
      refreshMetrics();
    } catch (err) {
      console.error('Batch error:', err);
      showNotice(`Batch analysis unavailable: ${err.message}`);
    } finally {
      btnRunBatch.disabled = false;
      btnRunBatch.innerHTML = `<span>🚀</span> Execute Batch Inference`;
    }
  });

  function renderBatchTable(items) {
    const filter = batchFilter ? batchFilter.value : 'all';
    const filtered = items.filter(item => {
      if (filter === 'all') return true;
      return item.sentiment.toLowerCase() === filter;
    });

    if (filtered.length === 0) {
      batchTableBody.innerHTML = '';
      batchEmpty.style.display = 'block';
      return;
    }

    batchEmpty.style.display = 'none';
    batchTableBody.innerHTML = filtered.map((item, idx) => {
      const isPos = item.sentiment.toLowerCase() === 'positive';
      const confPct = Math.round(item.confidence * 100);
      return `
        <tr>
          <td><strong style="color: var(--text-dim);">#${idx + 1}</strong></td>
          <td style="max-width: 420px; white-space: normal;">${escapeHtml(item.text)}</td>
          <td>
            <span class="badge-sentiment-table ${isPos ? 'positive' : 'negative'}">
              ${isPos ? '● Positive' : '● Negative'}
            </span>
          </td>
          <td><strong>${confPct}%</strong></td>
          <td style="font-family: var(--font-mono); color: var(--neutral); font-size: 0.8rem;">
            ${item.latency_ms} ms
          </td>
        </tr>
      `;
    }).join('');
  }

  if (batchFilter) {
    batchFilter.addEventListener('change', () => {
      renderBatchTable(batchStore);
    });
  }

  if (btnExportBatch) {
    btnExportBatch.addEventListener('click', () => {
      if (batchStore.length === 0) {
        alert('No batch predictions available to export.');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(batchStore, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute('href', dataStr);
      downloadAnchor.setAttribute('download', `sentiment_batch_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  // -------------------------------------------------------------------------
  // 4. MLOps Telemetry & CloudWatch Metrics
  // -------------------------------------------------------------------------
  async function refreshMetrics() {
    try {
      const res = await fetch('/metrics');
      if (!res.ok) return;
      const data = await res.json();

      statTotalReq.textContent = data.total_requests.toLocaleString();
      statTotalPred.textContent = data.total_predictions.toLocaleString();
      statAvgLatency.textContent = `${data.average_latency_ms} ms`;

      // Format uptime
      const s = Math.floor(data.uptime_seconds);
      const mins = Math.floor(s / 60);
      const secs = s % 60;
      statUptime.textContent = `${mins}m ${secs}s`;

      // Pos / Neg distribution
      const pos = data.positive_predictions || 0;
      const neg = data.negative_predictions || 0;
      const total = pos + neg;

      if (total > 0) {
        const posRatio = Math.round((pos / total) * 100);
        const negRatio = 100 - posRatio;
        statPosNegRatio.textContent = `${posRatio}% / ${negRatio}%`;

        splitPositive.style.width = `${posRatio}%`;
        splitNegative.style.width = `${negRatio}%`;

        legendPos.textContent = `Positive: ${pos} (${posRatio}%)`;
        legendNeg.textContent = `Negative: ${neg} (${negRatio}%)`;
      } else {
        statPosNegRatio.textContent = '0% / 0%';
        splitPositive.style.width = '50%';
        splitNegative.style.width = '50%';
        legendPos.textContent = 'Positive: 0';
        legendNeg.textContent = 'Negative: 0';
      }
    } catch (err) {
      console.warn('Could not fetch metrics:', err);
    }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      if (isPolling) {
        refreshMetrics();
      }
    }, 3500);
  }

  if (togglePolling) {
    togglePolling.addEventListener('click', () => {
      isPolling = !isPolling;
      togglePolling.textContent = isPolling ? 'Live · every 3.5s' : 'Live updates paused';
    });
  }

  // -------------------------------------------------------------------------
  // 5. System Health Beacon Check
  // -------------------------------------------------------------------------
  async function checkSystemHealth() {
    const healthBeacon = document.getElementById('systemHealthBeacon');
    const healthText = document.getElementById('healthStatusText');
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        healthText.textContent = data.status === 'healthy' ? 'Service is healthy' : 'Service is degraded';
        healthBeacon.classList.toggle('is-healthy', data.status === 'healthy');
        healthBeacon.classList.toggle('is-offline', data.status !== 'healthy');
        const env = document.getElementById('runtimeEnvironment');
        if (env) env.textContent = data.environment || 'Unknown';
        if (data.model_version) {
          modelVersionBadge.textContent = `v${data.model_version}`;
          const architectureVersion = document.getElementById('architectureVersion');
          if (architectureVersion) architectureVersion.textContent = `v${data.model_version}`;
        }
      } else {
        healthText.textContent = 'Service is degraded';
        healthBeacon.classList.remove('is-healthy');
        healthBeacon.classList.add('is-offline');
      }
    } catch {
      healthText.textContent = 'Service unavailable';
      healthBeacon.classList.remove('is-healthy');
      healthBeacon.classList.add('is-offline');
    }
  }

  // -------------------------------------------------------------------------
  // 6. Drift Detection Simulator
  // -------------------------------------------------------------------------
  btnSimulateDrift.addEventListener('click', async () => {
    driftConsole.textContent = 'Analyzing varied sample reviews…';
    btnSimulateDrift.disabled = true;

    const testPayloads = [
      "ksjdalkf jwioejf awoeijfoawij", // out of domain gibberish
      "this thing was like idk tbh not really whatever lol", // slang drift
      "great product but wait actually horrible and terrible scam", // conflicting features
      "magnifique incroyable fantastique cinema", // multilingual drift
      "neutral baseline sentence without distinct polarity words",
    ];

    try {
      driftConsole.textContent += ` Sending ${testPayloads.length} sample reviews to the model…`;
      const res = await fetch('/predict/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: testPayloads }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const batchData = await res.json();

      let confSum = 0;
      batchData.results.forEach(r => confSum += r.confidence);
      const avgConfidence = confSum / batchData.results.length;

      driftConsole.textContent = `Sample analysis complete. Average model confidence: ${Math.round(avgConfidence * 100)}%. This is a demonstration over ${batchData.results.length} examples, not a drift measurement.`;

      refreshMetrics();
    } catch (err) {
      driftConsole.textContent = `Sample analysis failed: ${err.message}`;
    } finally {
      btnSimulateDrift.disabled = false;
    }
  });

  // Live public headline feed
  const liveFeedType = document.getElementById('liveFeedType');
  const refreshLiveFeed = document.getElementById('refreshLiveFeed');
  const liveFeedItems = document.getElementById('liveFeedItems');
  const liveFeedStatus = document.getElementById('liveFeedStatus');

  async function loadLiveFeed() {
    if (!liveFeedType || !liveFeedItems || !liveFeedStatus) return;
    refreshLiveFeed.disabled = true;
    liveFeedStatus.textContent = 'Fetching headlines and analyzing sentiment...';
    liveFeedItems.replaceChildren();
    try {
      const response = await fetch(`/live-feed?feed=${encodeURIComponent(liveFeedType.value)}&limit=12`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Feed request failed (${response.status})`);
      liveFeedStatus.textContent = `${data.items.length} headlines | ${data.source} | refreshed ${new Date(data.analyzed_at).toLocaleTimeString()}`;
      data.items.forEach(item => {
        const card = document.createElement('article');
        card.className = 'live-headline-card';
        const title = document.createElement('h3');
        title.textContent = item.title;
        const summary = document.createElement('p');
        summary.textContent = item.summary;
        const meta = document.createElement('div');
        meta.className = 'live-headline-meta';
        const sentiment = document.createElement('span');
        sentiment.className = `badge-sentiment-table ${item.sentiment.toLowerCase()}`;
        sentiment.textContent = `${item.sentiment} | ${Math.round(item.confidence * 100)}% confidence`;
        const source = document.createElement('span');
        source.textContent = item.source || item.published;
        meta.append(sentiment, source);
        card.append(title, summary, meta);
        if (item.url && /^https?:\/\//i.test(item.url)) {
          const link = document.createElement('a');
          link.href = item.url;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = 'Open original article';
          card.append(link);
        }
        liveFeedItems.append(card);
      });
      if (!data.items.length) liveFeedStatus.textContent = 'No headlines were returned. Try refreshing shortly.';
    } catch (error) {
      liveFeedStatus.textContent = `Live feed unavailable: ${error.message}`;
    } finally {
      refreshLiveFeed.disabled = false;
    }
  }
  if (refreshLiveFeed) refreshLiveFeed.addEventListener('click', loadLiveFeed);
  if (liveFeedType) liveFeedType.addEventListener('change', loadLiveFeed);
  document.querySelector('[data-tab="tab-live"]')?.addEventListener('click', loadLiveFeed);

  // Utilities
  function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;',
    }[tag] || tag));
  }

  function showNotice(message) {
    let notice = document.getElementById('appNotice');
    if (!notice) {
      notice = document.createElement('div');
      notice.id = 'appNotice';
      notice.setAttribute('role', 'status');
      notice.className = 'app-notice';
      document.querySelector('.main-area').prepend(notice);
    }
    notice.textContent = message;
    notice.classList.add('visible');
    window.clearTimeout(notice._hideTimer);
    notice._hideTimer = window.setTimeout(() => notice.classList.remove('visible'), 5000);
  }

  // Initial load
  checkSystemHealth();
  refreshMetrics();
  startPolling();
});
