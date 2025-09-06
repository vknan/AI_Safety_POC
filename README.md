# AI_Safety_POC

# POC Roadmap (10 Steps)

## 1) Define scope, success criteria, and safety guardrails

* **Tasks (from assignment):** Abuse Language Detection, Escalation Pattern Recognition (conversation-level), Crisis Intervention (self-harm/distress), Age-based Content Filtering.&#x20;
* **Latency target (CPU):** p50 < 60ms/message; p95 < 150ms/message.
* **Quality targets (per task):** minimum F1 ≥ 0.80 on held-out set; *for Crisis*, favor **recall** ≥ 0.92 with calibrated thresholds + human-in-the-loop.
* **Operational guardrails:** confidence thresholds, abstain/route-to-human, rate limits, privacy redaction, full audit logging.

## 2) Data strategy & labeling plan

* Use only **public, anonymized** datasets (your brief requires this). Curate subsets per task and augment with synthetic but *harmless* paraphrases for class balance. Note multilingual slang/emoji variants to test robustness. Document sources + licenses in README and the Technical Report.&#x20;

**Lightweight taxonomy (editable):**

| Task       | Labels                                                                                      | Notes                              |
| ---------- | ------------------------------------------------------------------------------------------- | ---------------------------------- |
| Abuse      | `toxic`, `threat`, `harassment`, `hate`, `sexual/minors`, `self-harm-encouragement`, `none` | Multi-label possible               |
| Escalation | `stable`, `rising`, `critical`                                                              | Computed over conversation windows |
| Crisis     | `none`, `distress`, `self-harm`                                                             | Tune for **high recall**           |
| Age Filter | `safe_all`, `13+`, `16+`, `18+`                                                             | Policy mapping configurable        |

## 3) Preprocessing & feature pipeline

* Normalize unicode/emoji, lowercase (except for proper nouns if needed), URL/user handle masking, optional PII redaction.
* Tokenization with a transformer tokenizer (for the neural model) + TF-IDF (for baselines).
* Language detection → route non-English to a multilingual model or fallback.

## 4) Baselines first (fast + explainable)

* **Abuse & Age Filter:** Logistic Regression or Linear SVM on TF-IDF n-grams.
* **Crisis:** Linear model but with **recall-oriented thresholding**; add phrase/rule fallbacks for extreme terms to avoid misses.
* **Escalation:** Conversation window features: slope of negativity/toxicity scores, moving average deltas, turn cadence. Start with tree-based model (XGBoost/LightGBM) on aggregated features.

> Rationale: establishes a CPU-friendly, easy-to-explain baseline you can demo immediately; you’ll then optionally swap in a small transformer for accuracy.

## 5) Neural upgrade (optional but recommended)

* Swap baselines with a compact transformer (e.g., DistilBERT-class) fine-tuned per task.
* For **Escalation**, use a lightweight sequence model over turns (e.g., a transformer encoder across the last *k* messages) or compute per-turn scores + a small GRU/temporal-XGB.

## 6) Real-time simulation & integration

**Minimal architecture you can ship quickly:**

```
[Chat UI/CLI] → [Gateway API]
                     ↓
               [Inference Service]
               ├─ Abuse Classifier
               ├─ Crisis Classifier (recall-tuned)
               ├─ Age Filter
               └─ Escalation Module (over sliding window)
                     ↓
         [Policy/Rule Engine + Thresholds]
                     ↓
         [Actions: allow/warn/filter/escalate]
                     ↓
          [Audit Log + Metrics + Traces]
```

* Keep models **modular** so you can hot-swap them.&#x20;

## 7) Evaluation & calibration

**Core metrics (per class)**

* Precision $P=\frac{TP}{TP+FP}$, Recall $R=\frac{TP}{TP+FN}$, F1 $=2\cdot\frac{PR}{P+R}$.
* AUROC/PR-AUC for threshold selection; **Calibrate** with Platt scaling or temperature scaling (store calibration params).
* **Fairness slices:** report by language, dialect, and any benign proxies you have (e.g., text length, emoji density).
* **Conversation metrics (Escalation):** window-level F1; time-to-detect (turns until `rising/critical`).

**Example eval table template:**

| Task       | Model                   | Macro-F1 | Recall (positive) | PR-AUC | p95 Latency (ms) |
| ---------- | ----------------------- | -------: | ----------------: | -----: | ---------------: |
| Abuse      | TF-IDF + LinearSVM      |     0.84 |              0.86 |   0.91 |               42 |
| Crisis     | DistilBERT (calibrated) |     0.88 |          **0.94** |   0.95 |               78 |
| Escalation | Temporal-XGB            |     0.82 |              0.83 |   0.88 |               65 |
| Age Filter | TF-IDF + LR             |     0.87 |              0.86 |   0.92 |               38 |

## 8) Safety, ethics, and human-in-the-loop

* **Bias mitigation:** balanced sampling, data augmentation, threshold per slice if needed, post-processing equalized thresholds.
* **Explainability:** show top n-grams for linear models; for transformers, use simple token attributions (e.g., gradients) with a disclaimer.
* **Crisis path:** any `self-harm/distress` above a low threshold → **immediately** redact + display crisis notice and trigger human review (simulated queue). Document this in the report and show in the video demo.&#x20;

## 9) Packaging & deliverables mapping

* **Code repo** with training, inference, integration, eval scripts; small embedded sample datasets + links; README with setup/run steps.
* **10-minute video**: architecture → why these models → tradeoffs → live demo → next steps.
* **2–4 page Technical Report**: design, data, training, results, leadership/iteration plan.&#x20;

## 10) Timeline (you can hit this in \~3–5 focused days)

Day 1: baselines + CLI demo → Day 2: transformer fine-tune + eval → Day 3: real-time API/stream → Day 4: polish UI + logging → Day 5: video + report.

---

# Example Repo Structure

```
ai-safety-poc/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ data/
│  │  ├─ sample/              # tiny CSVs for quick demo
│  │  └─ loaders.py
│  ├─ models/
│  │  ├─ abuse_baseline.py
│  │  ├─ crisis_baseline.py
│  │  ├─ age_baseline.py
│  │  ├─ escalation_temporal.py
│  │  └─ transformers_head.py
│  ├─ inference/
│  │  ├─ pipeline.py          # orchestrates all models per message
│  │  ├─ policy_engine.py     # thresholds + actions
│  │  └─ server.py            # FastAPI (CPU)
│  ├─ training/
│  │  ├─ train_abuse.py
│  │  ├─ train_crisis.py
│  │  ├─ train_age.py
│  │  └─ train_escalation.py
│  ├─ evaluation/
│  │  ├─ eval_classification.py
│  │  └─ fairness.py
│  └─ ui/
│     ├─ cli_demo.py
│     └─ web_demo.py          # Streamlit or simple React front
└─ docs/
   ├─ technical_report.pdf
   └─ video_link.txt
```

---

# Minimal Working Pieces (you can paste straight into files)

## A) Inference pipeline (FastAPI, CPU-friendly)

```python
# src/inference/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.pipeline import SafetyPipeline

app = FastAPI(title="AI Safety POC")
pipe = SafetyPipeline.load_default()  # loads lightweight models + thresholds

class Msg(BaseModel):
    conv_id: str
    user_id: str
    text: str
    ts: float

@app.post("/analyze")
def analyze(msg: Msg):
    result = pipe.run(msg)
    return result  # includes per-task scores, final action, reasons
```

```python
# src/inference/pipeline.py
from src.models.abuse_baseline import AbuseModel
from src.models.crisis_baseline import CrisisModel
from src.models.age_baseline import AgeModel
from src.models.escalation_temporal import EscalationModel
from src.inference.policy_engine import PolicyEngine
from collections import defaultdict

class SafetyPipeline:
    def __init__(self, abuse, crisis, age, escalation, policy):
        self.abuse = abuse
        self.crisis = crisis
        self.age = age
        self.escalation = escalation
        self.policy = policy
        self.history = defaultdict(list)  # conv_id -> [(ts, text, scores)]

    @classmethod
    def load_default(cls):
        return cls(
            abuse=AbuseModel.load(),
            crisis=CrisisModel.load(),
            age=AgeModel.load(),
            escalation=EscalationModel.load(),
            policy=PolicyEngine.load()
        )

    def run(self, msg):
        abuse_scores = self.abuse.predict_proba(msg.text)
        crisis_scores = self.crisis.predict_proba(msg.text)
        age_scores   = self.age.predict_proba(msg.text)

        # append to history and compute escalation over last k turns
        hist = self.history[msg.conv_id]
        hist.append((msg.ts, msg.text, {"abuse": abuse_scores, "crisis": crisis_scores}))
        last_k = hist[-6:]
        esc_label, esc_score = self.escalation.score(last_k)

        action, reasons = self.policy.decide(
            abuse=abuse_scores, crisis=crisis_scores, age=age_scores,
            escalation=(esc_label, esc_score)
        )
        return {
            "abuse": abuse_scores, "crisis": crisis_scores, "age": age_scores,
            "escalation": {"label": esc_label, "score": esc_score},
            "action": action, "reasons": reasons
        }
```

```python
# src/inference/policy_engine.py
import math

class PolicyEngine:
    @classmethod
    def load(cls):
        # thresholds configurable via env/JSON
        return cls(
            t_abuse=0.65,
            t_crisis=0.40,  # lower threshold: prefer recall
            t_age_18=0.60,
            t_escalation=0.70
        )

    def __init__(self, t_abuse, t_crisis, t_age_18, t_escalation):
        self.t_abuse = t_abuse
        self.t_crisis = t_crisis
        self.t_age_18 = t_age_18
        self.t_escalation = t_escalation

    def decide(self, abuse, crisis, age, escalation):
        reasons = []
        action = "allow"

        if max(crisis.values()) >= self.t_crisis:
            action = "escalate_to_human"
            reasons.append("possible_crisis")

        if max(abuse.values()) >= self.t_abuse:
            action = "warn_and_filter"
            reasons.append("abusive_language")

        if age.get("18+", 0.0) >= self.t_age_18:
            action = "age_block"
            reasons.append("age_inappropriate")

        if escalation[1] >= self.t_escalation:
            reasons.append(f"escalation:{escalation[0]}")

        return action, reasons
```

## B) Baseline classifier (TF-IDF + LinearSVM)

```python
# src/models/abuse_baseline.py
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

class AbuseModel:
    def __init__(self, vec, clf, labels):
        self.vec, self.clf, self.labels = vec, clf, labels

    @classmethod
    def load(cls, path="models/abuse_baseline.joblib"):
        if os.path.exists(path):
            vec, clf, labels = joblib.load(path)
            return cls(vec, clf, labels)
        # tiny in-memory dummy for first run
        labels = ["toxic","threat","harassment","hate","sexual/minors","none"]
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        clf = CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
        # NOTE: in real training, fit vec+clf and save; here return unfitted placeholders
        return cls(vec, clf, labels)

    def predict_proba(self, text: str):
        # In POC, return mocked probabilities until trained; replace with real infer
        return {l: 0.0 for l in self.labels} | {"toxic": 0.72}  # example
```

> You’ll duplicate this pattern for **Crisis** and **Age** (with their own labels), and build a small temporal model for **Escalation** that reads the last *k* messages and emits (`stable|rising|critical`, score).

## C) Training & evaluation skeleton

```python
# src/training/train_abuse.py
import joblib, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd, numpy as np

def load_data():
    df = pd.read_csv("src/data/sample/abuse_train.csv")  # tiny, anonymized sample
    return df["text"].tolist(), df["label"].tolist()

X, y = load_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
Xtrv, Xtev = vec.fit_transform(Xtr), vec.transform(Xte)

clf = CalibratedClassifierCV(base_estimator=LinearSVC(), cv=3)
clf.fit(Xtrv, ytr)

pred = clf.predict(Xtev)
print(classification_report(yte, pred, digits=3))
joblib.dump((vec, clf, sorted(set(y))), "models/abuse_baseline.joblib")
```

---

# Escalation detection: simple but effective approach

**Feature recipe (per conversation window of last k=6 turns):**

* mean/max per-turn toxicity, **delta** of toxicity across turns, rolling variance
* count of aggressive terms, frequency of second-person imperatives
* turn interval shrinking? (faster back-and-forth can correlate with escalation)

**Model options:**

* Start with **XGBoost/LightGBM** on these aggregations.
* Upgrade to a tiny transformer over the 6 turns (CLS pooled) if needed.

---

# “Try it yourself” (quick demo loop)

### 1) Run the API

```bash
uvicorn src.inference.server:app --reload --port 8000
```

### 2) Send messages

```bash
curl -X POST http://localhost:8000/analyze \
 -H "Content-Type: application/json" \
 -d '{"conv_id":"c1","user_id":"u1","text":"you are worthless, go hurt yourself","ts": 1730900000.0}'
```

**Expect:** `action` likely `escalate_to_human`, `reasons` include `possible_crisis` and maybe `abusive_language`.

---

# Evaluation & math (concise)

For each positive class $c$:

$$
\text{Precision}_c=\frac{TP_c}{TP_c+FP_c},\quad
\text{Recall}_c=\frac{TP_c}{TP_c+FN_c},\quad
\text{F1}_c=2\cdot\frac{\text{Precision}_c\cdot\text{Recall}_c}{\text{Precision}_c+\text{Recall}_c}
$$

**Macro-F1** is the unweighted mean across classes. **Calibrate** decision thresholds by maximizing F1 or setting a target recall (e.g., Crisis ≥ 0.92), then record p50/p95 latency on CPU.

---

# Bias, ethics & governance (what to show in the video)

* Show **slice metrics** (e.g., by language or text length) to illustrate fairness checks.
* Make **thresholds configurable** (JSON or UI sliders) to demonstrate trade-offs (accuracy vs. speed/false positives).
* Implement **abstain** (return “needs human review”) when confidence ∈ $[0.4,0.6]$ band.
* Log **all** decisions with hashed user IDs, timestamps, inputs after redaction, model scores, and action; export to CSV/Parquet for audit.

---

# What to record in the 10-minute video (script outline)

1. **Intro & goals** (30s): 4 safety tasks, CPU, real-time.&#x20;
2. **Architecture** (2m): walk through the pipeline and why modular.
3. **Models & trade-offs** (3m): baselines → transformer upgrade; latency vs. accuracy; crisis recall priority.
4. **Live demo** (3m): benign → abusive; escalating conversation; crisis trigger; age filter block; show logs.
5. **Ethics & next steps** (1.5m): bias slices, calibration, human-in-the-loop; production hardening.

---

# 2–4 page Technical Report (bullet skeleton you can expand)

* **Design decisions:** why per-task models + rule/policy layer; CPU targets.
* **Data & preprocessing:** sources, splits, augmentations, normalization, language routing.
* **Models & training:** baselines + hyperparams; transformer choice; calibration method.
* **Results:** table with metrics + latency; fairness slices; examples of correct & incorrect cases.
* **Leadership & iteration:** roadmap for hardening (CI/CD, canarying, red-team tests), labeling ops, privacy reviews.&#x20;

---

# Diagram & media placeholders (drop into README/report)

* **\[Diagram Placeholder]** System Pipeline: (place the ASCII architecture diagram above or a simple block diagram).
* **\[Image Placeholder]** Confusion matrices per task.
* **\[Video Placeholder]** Link to your 10-min walkthrough.&#x20;

---

# Real-world examples to test in your demo

1. **Benign:** “Let’s meet at 5?” → allow.
2. **Borderline slang:** “That take is trash lol” → warn? (tune thresholds).
3. **Escalation pattern:** neutral → sarcasm → insult → threat → **critical**.
4. **Crisis cue:** “I can’t go on anymore.” → immediate escalate\_to\_human.
5. **Age filter:** “NSFW link …” with user age < 18 → age\_block.

---

# Next steps (if you have extra time)

* Swap in a compact multilingual model; add **language-aware** thresholds.
* Add **streaming mode** (websocket) for true turn-by-turn scoring.
* Implement **explanations** pane (top tokens/sentences influencing the decision).

---

If you’d like, I can generate a starter repo (folders/files pre-created with these scripts), plus a one-page checklist you can print while recording your video.
