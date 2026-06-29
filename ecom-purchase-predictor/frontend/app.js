/**
 * PurchaseIQ — app.js
 * ─────────────────────────────────────────────────────────────
 * Handles CSV upload, in-browser ML simulation, charts & table.
 *
 * NOTE: The model scoring functions (rfPredict, dtPredict, lrPredict)
 * simulate your trained sklearn models using weighted logic derived from
 * feature importance observed during training.
 *
 * To connect your REAL .pkl models:
 *  1. Add a Flask/FastAPI backend (see ml/train.py)
 *  2. Replace the runPredictions() call with a fetch() to /api/predict
 *     and pass the rows as JSON.
 * ─────────────────────────────────────────────────────────────
 */

'use strict';

// ── Chart registry (so we can destroy before redrawing) ───────────────────
const CHARTS = {};

// ── Drop Zone Setup ───────────────────────────────────────────────────────
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── CSV Parser ────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split('\n');
  return lines.slice(1).map((line) => {
    const v = line.split(',');
    return {
      age:      parseFloat(v[0]),
      session:  parseFloat(v[1]),
      pages:    parseInt(v[2]),
      cart:     parseInt(v[3]),
      days:     parseInt(v[4]),
      discount: parseInt(v[5]),
      purchased: parseInt(v[6]),
    };
  }).filter((r) => !isNaN(r.age) && !isNaN(r.purchased));
}

// ── File Handler ──────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a .csv file.');
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const rows = parseCSV(e.target.result);
      if (!rows.length) { alert('No valid rows found in CSV.'); return; }

      // Update drop zone UI
      dropZone.querySelector('.dz-title').textContent = `✓  ${file.name}`;
      dropZone.querySelector('.dz-sub').textContent   = `${rows.length.toLocaleString()} rows loaded successfully`;
      dropZone.style.borderColor = '#6c63ff';
      dropZone.style.background  = 'rgba(108,99,255,.06)';

      processData(rows, file.name);
    } catch (err) {
      alert('Could not parse CSV. Make sure the column order matches the required format.');
      console.error(err);
    }
  };
  reader.readAsText(file);
}

// ── ML Model Simulations ──────────────────────────────────────────────────
// Each returns a probability [0..1] of purchase.

function rfPredict(r) {
  // Random Forest — weights cart & pages heavily (ensemble behavior)
  let score = 0;
  score += r.cart    * 0.22;
  score += r.pages   * 0.018;
  score += r.session * 0.014;
  score += r.discount * 0.18;
  score -= r.days    * 0.008;
  score += (r.age > 25 && r.age < 55) ? 0.10 : 0;
  return clamp(score / 1.5);
}

function dtPredict(r) {
  // Decision Tree — hard threshold splits
  let score = 0;
  if (r.cart >= 3)     score += 0.40;
  if (r.pages >= 15)   score += 0.25;
  if (r.session >= 10) score += 0.15;
  if (r.discount)      score += 0.12;
  if (r.days <= 7)     score += 0.08;
  return clamp(score);
}

function lrPredict(r) {
  // Logistic Regression — linear combination with sigmoid
  const z = -1.4
    + r.cart    * 0.55
    + r.pages   * 0.045
    + r.session * 0.035
    + r.discount * 0.50
    - r.days    * 0.020
    + (r.age - 35) * 0.010;
  return sigmoid(z);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function clamp(v)   { return Math.max(0, Math.min(1, v)); }

// ── Accuracy Calculator ───────────────────────────────────────────────────
function calcAccuracy(rows, predictFn) {
  let correct = 0;
  rows.forEach((r) => {
    if ((predictFn(r) >= 0.5 ? 1 : 0) === r.purchased) correct++;
  });
  return Math.round(correct / rows.length * 100);
}

// ── Main Processing ───────────────────────────────────────────────────────
function processData(rows, fileName) {
  const n = rows.length;

  // ── Dataset stats
  const buys  = rows.filter((r) => r.purchased === 1);
  const nobuy = rows.filter((r) => r.purchased === 0);
  const avgSession = (rows.reduce((a, r) => a + r.session, 0) / n).toFixed(1);

  setText('file-name-title', `Dataset Overview — ${fileName}`);
  setText('file-sub', `${n.toLocaleString()} rows · ${buys.length.toLocaleString()} purchases · ${nobuy.length.toLocaleString()} non-purchases`);
  setText('st-total',      n.toLocaleString());
  setText('st-buy',        buys.length.toLocaleString());
  setText('st-buy-pct',    `${Math.round(buys.length / n * 100)}% of total`);
  setText('st-nobuy',      nobuy.length.toLocaleString());
  setText('st-nobuy-pct',  `${Math.round(nobuy.length / n * 100)}% of total`);
  setText('st-session',    avgSession);

  // ── Model predictions
  const rfProbs = rows.map(rfPredict);
  const dtProbs = rows.map(dtPredict);
  const lrProbs = rows.map(lrPredict);

  const rfAvg = avg(rfProbs);
  const dtAvg = avg(dtProbs);
  const lrAvg = avg(lrProbs);

  const rfAcc = calcAccuracy(rows, rfPredict);
  const dtAcc = calcAccuracy(rows, dtPredict);
  const lrAcc = calcAccuracy(rows, lrPredict);

  setModelCard('rf', rfAvg, rfAcc);
  setModelCard('dt', dtAvg, dtAcc);
  setModelCard('lr', lrAvg, lrAcc);

  // ── Consensus
  const votes   = [rfAvg, dtAvg, lrAvg].filter((p) => p >= 0.5).length;
  const avgAcc  = Math.round((rfAcc + dtAcc + lrAcc) / 3);
  const bestAcc = Math.max(rfAcc, dtAcc, lrAcc);
  const bestName = rfAcc === bestAcc ? 'Random Forest' : dtAcc === bestAcc ? 'Decision Tree' : 'Logistic Regression';
  setText('cons-text',
    `${votes}/3 models predict customers are ${votes >= 2 ? 'likely' : 'unlikely'} to purchase · ` +
    `Avg model accuracy: ${avgAcc}% · Best model: ${bestName} (${bestAcc}%)`
  );

  // ── Charts
  renderCharts(rows, buys, nobuy);

  // ── Prediction table (first 50 rows)
  renderTable(rows.slice(0, 50));

  // ── Show results section
  const resultsEl = document.getElementById('results');
  resultsEl.classList.remove('hidden');

  // Trigger staggered animations
  const animEls = resultsEl.querySelectorAll('.animate-in');
  animEls.forEach((el) => {
    el.style.animation = 'none';
    void el.offsetHeight;
    el.style.animation = '';
  });

  // Scroll to results
  setTimeout(() => {
    document.getElementById('model-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// ── Set Model Card ────────────────────────────────────────────────────────
function setModelCard(id, prob, accuracy) {
  const pct  = Math.round(prob * 100);
  const buy  = prob >= 0.5;
  const card = document.getElementById('card-' + id);

  card.classList.remove('buy', 'nobuy');
  card.classList.add(buy ? 'buy' : 'nobuy');

  setText('verdict-' + id, buy ? '↑ Likely to purchase' : '↓ Unlikely to purchase');
  setText('acc-' + id, `${accuracy}% accuracy`);
  setText('pct-' + id, pct + '%');

  setTimeout(() => {
    const bar = document.getElementById('bar-' + id);
    bar.style.width      = pct + '%';
    bar.style.background = buy ? '#34d399' : '#f87171';
  }, 100);
}

// ── Render Charts ─────────────────────────────────────────────────────────
function renderCharts(rows, buys, nobuy) {
  Object.values(CHARTS).forEach((c) => c && c.destroy());
  Object.keys(CHARTS).forEach((k) => delete CHARTS[k]);

  const tickColor  = '#7a8099';
  const gridColor  = 'rgba(255,255,255,0.05)';
  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: tickColor, font: { size: 11 } } },
      y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } } },
    },
  };

  // 1. Purchase distribution
  CHARTS.purchase = new Chart(document.getElementById('chartPurchase'), {
    type: 'bar',
    data: {
      labels: ['Did Not Purchase', 'Purchased'],
      datasets: [{
        data: [nobuy.length, buys.length],
        backgroundColor: ['#f87171', '#34d399'],
        borderRadius: 6,
        borderSkipped: 'bottom',
      }],
    },
    options: {
      ...baseOpts,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw.toLocaleString()} customers` } },
      },
    },
  });

  // 2. Cart size vs purchase rate
  const cartMap = {};
  rows.forEach((r) => {
    if (!cartMap[r.cart]) cartMap[r.cart] = { buy: 0, total: 0 };
    cartMap[r.cart].total++;
    if (r.purchased) cartMap[r.cart].buy++;
  });
  const cartKeys  = Object.keys(cartMap).sort((a, b) => +a - +b);
  const cartRates = cartKeys.map((k) => Math.round(cartMap[k].buy / cartMap[k].total * 100));
  CHARTS.cart = new Chart(document.getElementById('chartCart'), {
    type: 'bar',
    data: {
      labels: cartKeys.map((k) => `${k}`),
      datasets: [{
        data: cartRates,
        backgroundColor: '#6c63ff',
        borderRadius: 6,
        borderSkipped: 'bottom',
      }],
    },
    options: {
      ...baseOpts,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% purchase rate` } },
      },
      scales: {
        ...baseOpts.scales,
        y: {
          ...baseOpts.scales.y,
          max: 100,
          ticks: { ...baseOpts.scales.y.ticks, callback: (v) => v + '%' },
        },
        x: { ...baseOpts.scales.x, title: { display: true, text: 'Items in cart', color: tickColor, font: { size: 11 } } },
      },
    },
  });

  // 3. Session duration histogram (10-min buckets)
  const bins      = Array(6).fill(0);
  const binLabels = ['0–10', '10–20', '20–30', '30–40', '40–50', '50–60'];
  rows.forEach((r) => {
    const b = Math.min(5, Math.floor(r.session / 10));
    bins[b]++;
  });
  CHARTS.session = new Chart(document.getElementById('chartSession'), {
    type: 'bar',
    data: {
      labels: binLabels,
      datasets: [{
        data: bins,
        backgroundColor: '#fb923c',
        borderRadius: 6,
        borderSkipped: 'bottom',
      }],
    },
    options: {
      ...baseOpts,
      scales: {
        ...baseOpts.scales,
        x: { ...baseOpts.scales.x, title: { display: true, text: 'Session duration (min)', color: tickColor, font: { size: 11 } } },
      },
    },
  });

  // 4. Discount impact
  const discYes = rows.filter((r) => r.discount === 1);
  const discNo  = rows.filter((r) => r.discount === 0);
  const dyRate  = discYes.length ? Math.round(discYes.filter((r) => r.purchased).length / discYes.length * 100) : 0;
  const dnRate  = discNo.length  ? Math.round(discNo.filter((r) => r.purchased).length  / discNo.length  * 100) : 0;
  CHARTS.discount = new Chart(document.getElementById('chartDiscount'), {
    type: 'bar',
    data: {
      labels: ['No Discount', 'Discount Used'],
      datasets: [{
        data: [dnRate, dyRate],
        backgroundColor: ['#a78bfa', '#34d399'],
        borderRadius: 6,
        borderSkipped: 'bottom',
      }],
    },
    options: {
      ...baseOpts,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% purchase rate` } },
      },
      scales: {
        ...baseOpts.scales,
        y: {
          ...baseOpts.scales.y,
          max: 100,
          ticks: { ...baseOpts.scales.y.ticks, callback: (v) => v + '%' },
        },
      },
    },
  });
}

// ── Render Table ──────────────────────────────────────────────────────────
function renderTable(sample) {
  let correct = 0;
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  sample.forEach((r, i) => {
    const pred  = rfPredict(r) >= 0.5 ? 1 : 0;
    const match = pred === r.purchased;
    if (match) correct++;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--muted)">${i + 1}</td>
      <td>${r.age}</td>
      <td>${r.session.toFixed(1)}</td>
      <td>${r.pages}</td>
      <td>${r.cart}</td>
      <td>${r.days}</td>
      <td>${r.discount ? 'Yes' : 'No'}</td>
      <td><span class="badge ${r.purchased ? 'badge-buy' : 'badge-nobuy'}">${r.purchased ? 'Buy' : 'No buy'}</span></td>
      <td><span class="badge ${pred        ? 'badge-buy' : 'badge-nobuy'}">${pred        ? 'Buy' : 'No buy'}</span></td>
      <td><span class="badge ${match ? 'badge-match' : 'badge-nomatch'}">${match ? '✓' : '✗'}</span></td>
    `;
    tbody.appendChild(tr);
  });

  setText('pred-acc-label', `RF accuracy on sample: ${Math.round(correct / sample.length * 100)}%`);
}

// ── Helpers ───────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function avg(arr) { return arr.reduce((a, v) => a + v, 0) / arr.length; }

// ── Animate results sections when they appear ─────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate-in');
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.stat-card, .model-card, .chart-card, .model-info-card')
  .forEach((el) => observer.observe(el));