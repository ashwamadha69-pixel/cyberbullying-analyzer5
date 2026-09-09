// ============================================================
// Cyberbullying Impact Analyzer — fully client-side inference
// Replicates the Python training pipeline exactly:
//   clean text -> tokenize -> TF-IDF (L2-normalized) -> logistic regression (softmax)
// ============================================================

const CATEGORY_WEIGHTS = {
  not_cyberbullying: 0.0,
  other_cyberbullying: 0.55,
  age: 0.6,
  gender: 0.75,
  religion: 0.85,
  ethnicity: 0.9,
};

const INTENSIFIER_WORDS = [
  "kill", "die", "hate", "ugly", "worthless", "stupid", "idiot",
  "disgusting", "pathetic", "loser", "never", "everyone", "always",
];

function cleanText(text) {
  text = String(text).toLowerCase();
  text = text.replace(/http\S+|www\S+/g, " ");
  text = text.replace(/@\w+/g, " ");
  text = text.replace(/#/g, " ");
  text = text.replace(/[^a-z\s]/g, " ");
  text = text.replace(/\s+/g, " ").trim();
  return text;
}

function tokenize(cleaned) {
  return cleaned.split(" ").filter((t) => t.length >= 2);
}

function predict(rawText) {
  const cleaned = cleanText(rawText);
  const tokens = tokenize(cleaned);

  // term frequency over vocabulary
  const tf = new Map();
  for (const tok of tokens) {
    const idx = MODEL_DATA.vocab[tok];
    if (idx !== undefined) {
      tf.set(idx, (tf.get(idx) || 0) + 1);
    }
  }

  // raw tf-idf values for present features, and L2 norm
  let sumSquares = 0;
  const tfidf = new Map();
  for (const [idx, count] of tf.entries()) {
    const val = count * MODEL_DATA.idf[idx];
    tfidf.set(idx, val);
    sumSquares += val * val;
  }
  const norm = Math.sqrt(sumSquares) || 1;

  const nClasses = MODEL_DATA.classes.length;
  const logits = new Array(nClasses).fill(0);
  for (let c = 0; c < nClasses; c++) {
    let dot = 0;
    const coefRow = MODEL_DATA.coef[c];
    for (const [idx, val] of tfidf.entries()) {
      dot += (val / norm) * coefRow[idx];
    }
    logits[c] = dot + MODEL_DATA.intercept[c];
  }

  // softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;

  return {
    label: MODEL_DATA.classes[bestIdx],
    confidence: probs[bestIdx],
    probs: MODEL_DATA.classes.map((cls, i) => ({ label: cls, prob: probs[i] })),
    tokenCount: tokens.length,
  };
}

function computeImpactScore(label, confidence, rawText) {
  const base = CATEGORY_WEIGHTS[label] ?? 0.5;
  const lower = rawText.toLowerCase();
  let hits = 0;
  for (const w of INTENSIFIER_WORDS) if (lower.includes(w)) hits++;
  const intensityBoost = Math.min(hits * 0.05, 0.2);

  let score = (base * 0.7 + confidence * 0.2 + intensityBoost) * 100;
  score = Math.max(0, Math.min(100, score));

  let tier;
  if (label === "not_cyberbullying") tier = "No significant concern";
  else if (score < 40) tier = "Low";
  else if (score < 65) tier = "Moderate";
  else if (score < 85) tier = "High";
  else tier = "Severe";

  return { score: Math.round(score * 10) / 10, tier };
}

function formatLabel(label) {
  return label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------- UI wiring ----------
const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBlock = document.getElementById("resultBlock");
const categoryOut = document.getElementById("categoryOut");
const confidenceOut = document.getElementById("confidenceOut");
const scoreOut = document.getElementById("scoreOut");
const tierOut = document.getElementById("tierOut");
const probList = document.getElementById("probList");
const explainOut = document.getElementById("explainOut");
const gaugeArc = document.getElementById("gaugeArc");
const gaugeNeedle = document.getElementById("gaugeNeedle");

const ARC_LENGTH = 251.2; // approx length of the half-circle path

function renderResult(rawText) {
  if (!rawText.trim()) return;
  const { label, confidence, probs } = predict(rawText);
  const { score, tier } = computeImpactScore(label, confidence, rawText);

  categoryOut.textContent = formatLabel(label);
  confidenceOut.textContent = (confidence * 100).toFixed(1) + "%";
  scoreOut.textContent = score;
  tierOut.textContent = tier;

  // gauge: 0-100 maps to 0deg-180deg rotation of needle from vertical-left(-90) baseline
  const angle = (score / 100) * 180 - 90;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  const offset = ARC_LENGTH - (score / 100) * ARC_LENGTH;
  gaugeArc.style.strokeDashoffset = offset;
  const arcColor = score < 40 ? "#5C7FC4" : score < 65 ? "#F9A66C" : score < 85 ? "#F96167" : "#D8474D";
  gaugeArc.style.stroke = arcColor;

  // probability breakdown, sorted descending
  const sorted = [...probs].sort((a, b) => b.prob - a.prob);
  probList.innerHTML = sorted
    .map(
      (p) => `
      <div class="prob-row">
        <span class="prob-name">${formatLabel(p.label)}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${(p.prob * 100).toFixed(1)}%"></div></div>
        <span class="prob-pct">${(p.prob * 100).toFixed(0)}%</span>
      </div>`
    )
    .join("");

  if (label === "not_cyberbullying") {
    explainOut.textContent =
      "This message doesn't show characteristics of cyberbullying based on the model's training data.";
  } else {
    explainOut.textContent = `This message shows characteristics of ${formatLabel(
      label
    ).toLowerCase()}-based cyberbullying, with an estimated ${tier.toLowerCase()} potential impact. Score combines category severity, model confidence, and language intensity.`;
  }

  resultBlock.classList.remove("hidden");
}

analyzeBtn.addEventListener("click", () => renderResult(textInput.value));
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) renderResult(textInput.value);
});

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    textInput.value = btn.dataset.sample;
    renderResult(btn.dataset.sample);
  });
});
