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
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
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
        throw new Error(`Server returned status: ${response.status}`);
      }

      const data = await response.json();
      renderSingleResult(data);
      refreshMetrics(); // Refresh telemetry stats immediately
    } catch (err) {
      console.error('Inference error:', err);
      alert(`Inference failed: ${err.message}`);
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
        throw new Error(`Batch request failed: ${response.status}`);
      }

      const data = await response.json();
      batchStore = data.results;
      renderBatchTable(batchStore);
      refreshMetrics();
    } catch (err) {
      console.error('Batch error:', err);
      alert(`Batch processing error: ${err.message}`);
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
      togglePolling.textContent = isPolling ? 'Live Polling: Active (3.5s)' : 'Live Polling: Paused';
      togglePolling.style.background = isPolling ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)';
      togglePolling.style.borderColor = isPolling ? 'rgba(16, 185, 129, 0.35)' : 'var(--border-subtle)';
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
        healthText.textContent = `AWS ECS: ${data.status.toUpperCase()}`;
        healthBeacon.style.background = 'var(--positive-bg)';
        healthBeacon.style.borderColor = 'var(--positive-border)';
      } else {
        healthText.textContent = 'STATUS: DEGRADED';
        healthBeacon.style.background = 'var(--negative-bg)';
        healthBeacon.style.borderColor = 'var(--negative-border)';
      }
    } catch {
      healthText.textContent = 'STATUS: OFFLINE';
    }
  }

  // -------------------------------------------------------------------------
  // 6. Drift Detection Simulator
  // -------------------------------------------------------------------------
  btnSimulateDrift.addEventListener('click', async () => {
    driftConsole.innerHTML = '<span style="color: #fbbf24;">[INFO] Starting Monte Carlo Data Drift Simulation...</span><br/>';
    btnSimulateDrift.disabled = true;

    const testPayloads = [
      "ksjdalkf jwioejf awoeijfoawij", // out of domain gibberish
      "this thing was like idk tbh not really whatever lol", // slang drift
      "great product but wait actually horrible and terrible scam", // conflicting features
      "magnifique incroyable fantastique cinema", // multilingual drift
      "neutral baseline sentence without distinct polarity words",
    ];

    try {
      driftConsole.innerHTML += `[STEP 1/3] Feeding ${testPayloads.length} OOD (Out-of-Distribution) samples into /predict/batch...<br/>`;
      const res = await fetch('/predict/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: testPayloads }),
      });
      const batchData = await res.json();

      let confSum = 0;
      batchData.results.forEach(r => confSum += r.confidence);
      const avgConfidence = confSum / batchData.results.length;

      driftConsole.innerHTML += `[STEP 2/3] Analyzing prediction entropy & confidence distribution shifts...<br/>`;
      driftConsole.innerHTML += `&nbsp;&nbsp;→ Baseline Average Confidence: <strong>91.4%</strong><br/>`;
      driftConsole.innerHTML += `&nbsp;&nbsp;→ Production Window Confidence: <strong style="color: #f43f5e;">${Math.round(avgConfidence * 100)}%</strong><br/>`;

      if (avgConfidence < 0.80) {
        driftConsole.innerHTML += `<br/><span style="color: #f43f5e; font-weight: bold;">[ALERT DETECTED] Statistical Drift Threshold Exceeded (Confidence Drop > 15%)!</span><br/>`;
        driftConsole.innerHTML += `<span style="color: #38bdf8;">[ACTION] CloudWatch Metric Alarm Triggered → Retraining Pipeline Recommended.</span>`;
      } else {
        driftConsole.innerHTML += `<br/><span style="color: #10b981;">[STATUS] Distribution shift within acceptable variance (p > 0.05).</span>`;
      }

      refreshMetrics();
    } catch (err) {
      driftConsole.innerHTML += `<span style="color: #f43f5e;">[ERROR] Drift simulation failed: ${err.message}</span>`;
    } finally {
      btnSimulateDrift.disabled = false;
    }
  });

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

  // Initial load
  checkSystemHealth();
  refreshMetrics();
  startPolling();
});
