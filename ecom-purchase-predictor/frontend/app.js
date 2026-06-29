/**
 * PurchaseIQ — Frontend Prediction Logic
 * ----------------------------------------
 * Simulates the three ML models using weighted scoring based on feature
 * importance observed during training. For a real deployment, replace
 * runModels() with a fetch() call to a Flask/FastAPI backend.
 */

let discountValue = 1;

function setDiscount(val) {
  discountValue = val;
  document.getElementById('discount-yes').classList.toggle('active', val === 1);
  document.getElementById('discount-no').classList.toggle('active',  val === 0);
}

// ── Feature reading ────────────────────────────────────────────────────────
function getInputs() {
  return {
    age:     parseFloat(document.getElementById('age').value)     || 0,
    session: parseFloat(document.getElementById('session').value) || 0,
    pages:   parseFloat(document.getElementById('pages').value)   || 0,
    cart:    parseFloat(document.getElementById('cart').value)    || 0,
    days:    parseFloat(document.getElementById('days').value)    || 0,
    discount: discountValue,
  };
}

// ── Model simulations ──────────────────────────────────────────────────────
/**
 * Each function returns a probability [0..1] of purchase.
 * Weights are illustrative — replace with real model outputs from your backend.
 */

function randomForest({ age, session, pages, cart, days, discount }) {
  // RF tends to weight cart & pages heavily
  let score = 0;
  score += cart    * 0.22;
  score += pages   * 0.018;
  score += session * 0.014;
  score += discount * 0.18;
  score -= days    * 0.008;
  score += (age > 25 && age < 55) ? 0.10 : 0;
  return clamp(score / 1.5);
}

function decisionTree({ age, session, pages, cart, days, discount }) {
  // DT uses hard threshold-style logic
  let score = 0;
  if (cart >= 3)    score += 0.40;
  if (pages >= 15)  score += 0.25;
  if (session >= 10) score += 0.15;
  if (discount)     score += 0.12;
  if (days <= 7)    score += 0.08;
  return clamp(score);
}

function logisticRegression({ age, session, pages, cart, days, discount }) {
  // LR uses a linear combination with sigmoid
  const z = -1.4
    + cart    * 0.55
    + pages   * 0.045
    + session * 0.035
    + discount * 0.50
    - days    * 0.020
    + (age - 35) * 0.010;
  return sigmoid(z);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function clamp(v) { return Math.max(0, Math.min(1, v)); }

function runModels(inputs) {
  return {
    rf: randomForest(inputs),
    dt: decisionTree(inputs),
    lr: logisticRegression(inputs),
  };
}

// ── UI update ──────────────────────────────────────────────────────────────
function setResult(id, badgeClass, prob) {
  const buy    = prob >= 0.5;
  const label  = buy ? '✓ Will Purchase' : '✗ Will Not Purchase';
  const pct    = Math.round(prob * 100);
  const card   = document.getElementById(`result-${id}`);
  const verdict = document.getElementById(`verdict-${id}`);
  const conf   = document.getElementById(`conf-${id}`);

  card.classList.remove('buy', 'no-buy');
  card.classList.add(buy ? 'buy' : 'no-buy');
  verdict.textContent = label;
  verdict.className = `verdict ${buy ? 'buy' : 'no-buy'}`;
  conf.textContent = `Model confidence: ${pct}%`;

  // confidence bar
  conf.innerHTML = `
    <div style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px">
        <span>Confidence</span><span>${pct}%</span>
      </div>
      <div style="background:rgba(255,255,255,.07);border-radius:99px;height:5px;overflow:hidden">
        <div style="width:${pct}%;height:100%;border-radius:99px;background:${buy ? 'var(--green)' : 'var(--red)'};transition:width 0.5s ease"></div>
      </div>
    </div>`;
}

function predict() {
  const btn = document.getElementById('btn-label');
  btn.textContent = 'Running…';

  // small delay for perceived loading feel
  setTimeout(() => {
    const inputs = getInputs();
    const probs  = runModels(inputs);

    setResult('rf', 'rf', probs.rf);
    setResult('dt', 'dt', probs.dt);
    setResult('lr', 'lr', probs.lr);

    // Consensus
    const votes    = [probs.rf, probs.dt, probs.lr].filter(p => p >= 0.5).length;
    const avgProb  = (probs.rf + probs.dt + probs.lr) / 3;
    const buyVotes = votes;

    const consensusCard = document.getElementById('consensus-card');
    const consensusText = document.getElementById('consensus-text');
    consensusCard.style.display = 'block';

    if (buyVotes === 3) {
      consensusText.textContent = `All 3 models agree: likely to purchase (avg ${Math.round(avgProb*100)}%)`;
    } else if (buyVotes === 2) {
      consensusText.textContent = `2 of 3 models predict: likely to purchase`;
    } else if (buyVotes === 1) {
      consensusText.textContent = `2 of 3 models predict: unlikely to purchase`;
    } else {
      consensusText.textContent = `All 3 models agree: unlikely to purchase (avg ${Math.round(avgProb*100)}%)`;
    }

    btn.textContent = 'Run Prediction';
  }, 350);
}

// Run on load so cards show blank state cleanly
document.addEventListener('DOMContentLoaded', () => {
  setDiscount(1);
});
