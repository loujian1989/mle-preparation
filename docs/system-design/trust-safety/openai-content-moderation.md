# OpenAI Unsafe Content Detection Pipeline — ML System Design

**Domain:** `trust-safety`
**Target Company:** OpenAI
**Difficulty Bar:** L6 (Staff)
**Date:** 2026-03-27
**Related Designs:** `stripe-fraud-detection.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Adversarial prompt evolution (new jailbreak techniques) partially addressed |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Adversarial robustness — the system must evolve faster than attacker innovation. Red-teaming cadence and policy update pipeline need explicit SLA.

---

## 1. Requirements

#### Functional Requirements
1. Detect and action unsafe content in real-time across modalities: text (prompts + completions), images, and video frames
2. Classify content into policy violation categories: CSAM, violence, self-harm, hate speech, illegal activity, deception/manipulation
3. Support configurable policy thresholds per deployment context (API vs. consumer product vs. enterprise)
4. Provide explainable decisions: policy category + confidence + SHAP-derived rationale for human review

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving, synchronous) | ≤ 100ms | Inline moderation on every API call; beyond 100ms, product latency degrades visibly |
| Availability | 99.99% | Moderation outage = unmoderated API traffic = policy/legal/reputational risk |
| Consistency | Strong (synchronous gate) | Content must be moderated *before* serving to user — eventual consistency not acceptable for safety-critical decisions |
| Throughput | ~200K peak QPS | ChatGPT + API combined traffic; scales with OpenAI's API growth |
| False positive rate | ≤ 0.1% on benign requests | False positives block legitimate users; trust degradation is a real churn signal |
| False negative rate | ≤ 0.01% on critical categories (CSAM) | Near-zero tolerance for CSAM misses; higher tolerance for edge-case hate speech |
| Bias parity | Demographic parity within ±5% | Moderation must not disproportionately flag content from specific demographic groups |

#### Scale Numbers (stated upfront)
- **API calls/day:** ~1B (ChatGPT + API + enterprise)
- **Peak QPS:** ~200K
- **Human review queue:** ~50K items/day requiring human judgment (0.005% of traffic)
- **Red team attack attempts:** ~10K/day (internal + external researchers)
- **Content categories:** 12 top-level policy categories; each with sub-categories

#### Out of Scope
- Watermarking AI-generated content (separate system)
- Copyright infringement detection (separate legal pipeline)
- Account-level ban decisions (trust & safety ops, not this system)
- Moderation of third-party plugins or tool outputs

> **OpenAI rubric:** Bias monitoring is a first-class requirement — not an afterthought. State demographic parity targets before designing the model. AI ethics discussion is real; have a formed position on policy calibration trade-offs.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| Text prompt + completion | `real-time` | request-time | API gateway → moderation service | Primary input; tokenized + embedded |
| Text embedding (prompt) | `real-time` | request-time | In-process text encoder (distilled BERT) | 768-d; encodes semantic meaning |
| Image (if multi-modal) | `real-time` | request-time | API payload | CLIP-encoded for vision classifiers |
| Conversation history (last 5 turns) | `real-time` | request-time | API session context | Context window for cumulative violation detection |
| User account tier | `static` | per-session | API auth → Redis | Enterprise accounts: different threshold config |
| Deployment context | `static` | per-request | API header | Consumer (strict) vs. API (configurable) vs. enterprise (custom) |
| Adversarial pattern library match | `batch` | 15 min | Red team findings → rules engine | Known jailbreak patterns: base64 encoding, role-play bypasses, indirect instruction |
| Historical violation rate (user) | `batch` | 1 hour | Spark on violation logs → Redis | Prior violators get stricter thresholds |

#### Label Definition
- **Labels (multi-class):** 12 policy categories (binary per category); one content item can violate multiple categories
- **Collection strategy:**
  - Human raters (Surge/Scale AI + internal RLHF labelers) rate sampled content
  - RLHF reward model: trained on pairwise comparisons of human rater decisions
  - Red team findings: adversarial examples labeled as violations → added to hard-negative training set
- **Label skew:** Severely imbalanced — most content (>99.9%) is benign; CSAM << hate speech << violence in frequency
- **Label delay:** Human review labels available within 24 hours of content submission; red team labels within 1 week
- **Bias risks:**
  - **Demographic bias:** Models trained on US English data may over-flag African American Vernacular English (AAVE) or non-Western cultural references → measure FPR by demographic proxy in evaluation
  - **Annotation bias:** Human raters disagree on edge cases (50% inter-rater agreement on borderline hate speech) → use RLHF to calibrate on disagreement cases; not majority vote
  - **Adversarial distribution shift:** Attackers actively shift content distribution to evade detection → retrain cadence must be faster than adversarial innovation cycle (target: 24hr red team → retrain SLA)

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Adversarial pattern library (rules) | Redis (key: pattern_hash → action) | Sub-ms lookup; updated every 15 min from red team findings |
| User violation history | Redis (TTL: 30 days; key: user_id) | Fast lookup; expiry prevents permanent over-flagging |
| Human review queue | PostgreSQL + internal queue system | Structured; supports reviewer assignment, audit trail, appeal workflow |
| Training data (labeled content) | S3 (Parquet, encrypted) | Content data requires encryption at rest; partition by date + label |
| RLHF annotation data | S3 (encrypted, access-controlled) | Sensitive; restricted access by policy team only |
| Model artifacts | S3 + versioned model registry | Each policy update = new model version; full audit trail |
| Violation logs | Kafka → Flink → BigQuery | Real-time streaming; analytics on policy category distribution |

#### Online vs. Offline Split

```
Offline (batch)                                    Online (synchronous, < 100ms)
────────────────────────────────────────────       ──────────────────────────────────────────
Human review labels → RLHF training data           Request: text prompt + completion
Red team findings → adversarial examples           Rules engine: pattern library match (<1ms)
Spark: bias audit (FPR by demographic proxy)       Text encoder: embed prompt (10ms, distilled BERT)
Daily retraining on new labeled data               Stage 1 classifier: text toxicity score (20ms)
Weekly red team → model update cycle               If score > low_threshold: Stage 2 detailed (30ms)
Bias parity evaluation before deployment           If score > high_threshold: block + queue for review
Champion/challenger: bias + accuracy gating        Async: log violation event to Kafka
```

**Two-stage classification (cascade):**
1. **Fast pre-filter** (< 5ms): rule-based pattern matching on known adversarial signatures; keyword hashing; regex on obfuscation patterns (base64, ROT13, leet-speak)
2. **Distilled BERT classifier** (< 20ms): 12 binary classifiers (one per policy category); small distilled model optimized for latency
3. **Full model** (< 50ms, only for borderline cases): larger BERT-based multi-class model with SHAP explanations; only triggered if Stage 2 score in [0.3, 0.7] (uncertain region)
4. **Human review queue** (async, < 24hr): cases where Stage 3 confidence < 0.85 AND category is non-critical; CSAM always routes to human review regardless of model confidence

#### Schema

```
ModerationRequest: {
  request_id:         string
  user_id:            string         # anonymized in logs
  deployment_context: enum[CONSUMER, API, ENTERPRISE]
  text_content:       string         # encrypted at rest
  image_url:          string?        # optional; for multi-modal
  timestamp:          timestamp
}

ModerationDecision: {
  request_id:         string
  action:             enum[ALLOW, BLOCK, REVIEW_QUEUE]
  violations:         ViolationCategory[]
  confidence:         float[]        # per category
  stage_triggered:    enum[RULES, STAGE1, STAGE2, STAGE3]
  shap_rationale:     string?        # top-3 features for human review
  latency_ms:         int
  timestamp:          timestamp
}

ViolationCategory: {
  category:           enum[CSAM, VIOLENCE, SELF_HARM, HATE_SPEECH, ILLEGAL, DECEPTION, ...]
  confidence:         float
  policy_version:     string         # which policy document this maps to
}
```

> **OpenAI rubric:** Bias risks stated upfront with mitigation. RLHF policy calibration tied to annotation disagreement. Human review is explicitly in the system design, not added as an afterthought.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Human review decisions + RLHF pairwise comparisons → S3; red team adversarial examples → S3
- **Feature engineering:**
  - Text tokenization: OpenAI BPE tokenizer (shared with GPT family); max 512 tokens
  - Adversarial augmentation: apply known obfuscation transforms (base64, rot13, char substitution) to training examples → train robustness to these transforms
  - Demographic proxy features (for bias audit only, not training input): detect potential demographic signals via proxy classifiers; use for evaluation FPR disaggregation only
- **Train/val/test split:** Time-based + adversarial-holdout; hold out all red team examples from the past 2 weeks as adversarial test set; ensure val/test have representation from all 12 categories
- **Orchestration:** Metaflow; RLHF reward model trained first; moderation classifier distills from reward model + supervised labels

#### Model Architecture

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Rule-based only (keyword filter) | < 1ms; deterministic | Easily evaded by paraphrase; high false positive rate | Stage 1 only (not sufficient alone) |
| Distilled BERT (chosen for Stage 2) | 20ms; good accuracy on known patterns; explainable via SHAP | Struggles with novel adversarial attacks; needs frequent retraining | **Chosen for Stage 2** |
| Full BERT / RoBERTa (chosen for Stage 3) | Highest accuracy; best SHAP explanations | 50ms; only used for borderline cases | **Chosen for Stage 3 (conditional)** |
| LLM-as-judge (GPT-4 for moderation) | Best nuanced understanding | > 500ms; 10× cost; not viable for inline path | Used for human review queue prioritization (async only) |

**RLHF Policy Calibration:**
- Reward model trained on pairwise human comparisons: "Is response A or B more policy-compliant?"
- Reward model score is used to calibrate threshold per policy category (not a fixed 0.5 threshold)
- Key trade-off: **strictness vs. false positives** — stricter thresholds block more harmful content but also block more legitimate use (researchers, medical professionals, fiction writers). OpenAI exposes this as a configurable parameter per deployment context

**Explainability (for human review):**
- SHAP values computed for Stage 3 decisions: top-3 token spans contributing to violation classification
- SHAP rationale included in human review queue item → reviewer sees which exact phrases triggered the flag
- Without SHAP: human reviewers cannot efficiently validate model decisions → throughput in review queue drops 3×

#### Training Infrastructure
- **Framework:** PyTorch + FSDP (full BERT base model); Hugging Face Transformers
- **Scale:** Stage 2 distilled model: 4× A10G GPUs, ~4hr/run; Stage 3 full model: 16× A100, ~12hr/run
- **Mixed precision:** bfloat16 for both stages
- **Eval metrics:** Per-category AUC, false positive rate (overall + by demographic proxy), false negative rate (weighted by severity: CSAM > violence > hate speech), latency p99 per stage

---

### 3b. Online Serving

#### Inference Path

```
API Request (text + optional image)
  → API Gateway (auth, rate limit)
  → Moderation Service (inline, synchronous)
      ├─ Stage 1: Rules Engine
      │    └─ Redis: adversarial pattern match (<1ms)
      │    → If matched: BLOCK immediately
      ├─ Stage 2: Distilled BERT Classifier
      │    └─ Tokenize + embed (5ms) → 12 binary classifiers (15ms)
      │    → If any category score > high_threshold: BLOCK
      │    → If all scores < low_threshold: ALLOW
      │    → If any score in [low, high]: escalate to Stage 3
      ├─ Stage 3 (conditional): Full BERT + SHAP
      │    └─ Full model inference (30ms) + SHAP computation (10ms)
      │    → If confidence > 0.85: take action (BLOCK or ALLOW)
      │    → If confidence < 0.85: ALLOW with REVIEW_QUEUE (async)
      └─ Response + async Kafka log (all decisions)
           + REVIEW_QUEUE if flagged for human review (PostgreSQL)
```

**Traffic routing:** ~90% of requests clear at Stage 1 or Stage 2; ~8% reach Stage 3; ~0.005% enter human review queue. Stage 3 is the latency-critical path — must be optimized first.

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Rules engine (Stage 1) | < 1ms | 1ms | Redis lookup; most requests pass through quickly |
| Stage 2 distilled BERT | 12ms | 20ms | TensorRT-optimized; 90% of requests stop here |
| Stage 3 full BERT + SHAP (10% of traffic) | 35ms | 60ms | Conditional; contributes ~6ms to average |
| Network + serialization | 5ms | 15ms | gRPC |
| **Total (p99, Stage 3 path)** | **~53ms avg** | **96ms** | Budget: ≤ 100ms ✓ |

#### Caching Strategy
- **Adversarial pattern cache (Redis):** Updated every 15 min from red team database; TTL: 20 min with refresh-ahead
- **No caching of content decisions:** Each request is unique; caching decisions would allow adversaries to probe the cache with slight variations to find allowed-but-unsafe content

---

### 3c. Monitoring

> **Designed upfront — bias monitoring is a first-class OpenAI requirement, not an afterthought.**

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Adversarial attack rate | Rolling count of Stage 1 matches | > 2× baseline sustained 1hr | Red team alert; pattern library update |
| False positive rate by demographic proxy | Daily disaggregated audit | FPR deviation > 5% across groups | Bias review; model recalibration |
| Per-category violation rate | Rolling 24hr vs. 7-day baseline | > 3× baseline on any category | Policy team alert; investigate source |
| Model score distribution (Stage 2) | Rolling mean ± 2σ | > 15% shift sustained 2hr | Shadow model comparison; check data pipeline |
| Human review queue depth | Real-time | > 24hr backlog (>50K items) | Scale reviewer capacity; check auto-routing |
| RLHF reward model agreement | Weekly evaluation on new human labels | < 85% pairwise agreement | Retrain reward model |
| p99 latency | Real-time APM | > 100ms sustained 3min | Circuit breaker; disable Stage 3; ALLOW borderline cases |

#### Shadow Scoring
- New policy version runs on 1% of traffic; decisions logged (not acted on); human review sample checked for accuracy
- Bias audit on shadow traffic before promotion: FPR parity across demographic proxies must be within ±5% of current champion
- Promotion requires: better AUC on critical categories (CSAM, violence) + no FPR regression + no latency regression

#### A/B Holdout Design
- **Unit:** Not standard A/B — policy changes affect all users simultaneously. Use **holdout slice by content category**: route 5% of borderline cases (Stage 3) to human-only decision; compare with model decision over 1 week
- **Primary metric:** False negative rate on human-validated ground truth (not predicted labels)
- **Guardrail:** False positive rate (user complaint rate as proxy), p99 latency

> **OpenAI rubric:** Bias monitoring cadence is daily with disaggregated FPR by demographic proxy. RLHF reward model is retrained when human agreement drops. This is designed *before* the model architecture, not after.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Stage 2/3 model server unavailable | No ML-based moderation | Retry ×1 (20ms timeout); circuit breaker | Rules-only (Stage 1); stricter thresholds on rule engine; ALLOW only clearly safe requests |
| Redis pattern library stale (> 30 min) | New adversarial patterns not caught | Monitor pattern library last-update timestamp; alert at 30 min | Stricter Stage 2 threshold (lower false negative rate trades for higher false positive rate) |
| Human review queue overflow | Backlog > 24hr SLA | Auto-prioritize by severity (CSAM > violence > others); alert policy ops team; scale reviewers | Temporarily block borderline cases (BLOCK instead of REVIEW_QUEUE) for critical categories |
| New jailbreak technique discovered | Stage 1 and Stage 2 miss novel attack | 24hr SLA: red team finding → pattern library update → model retrain; interim: manual monitoring of flagged sessions | Human review all Stage 3 borderline cases until pattern library updated |
| Bias drift detected (FPR imbalance) | Disproportionate blocking of specific groups | Immediate: recalibrate Stage 2 thresholds for affected category; 72hr: full model recalibration | Temporarily widen thresholds for affected category to reduce FPR while fix is deployed |
| RLHF reward model stale | Policy calibration drifts from intended policy | Weekly evaluation trigger; retrain if agreement < 85% | Current calibration stays; policy team manually reviews sampled decisions |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 0.5% 5xx → rules-only moderation; alert immediately (safety-critical)
- **Latency:** Trip at p99 > 100ms sustained 2min → disable Stage 3; borderline cases go to ALLOW + async review
- **Recovery:** Half-open after 15s (fastest recovery — safety system cannot be down long)

#### Degraded-Mode Behavior
1. **Level 1** — Full pipeline (Stage 1 + 2 + 3 + human review queue) — normal operation
2. **Level 2** — Stage 1 + Stage 2 only (no SHAP, no conditional Stage 3) — reduced accuracy on borderline cases
3. **Level 3** — Stage 1 rules only — lowest accuracy; stricter thresholds to compensate; critical categories still blocked

> **OpenAI rubric:** Safety system failure modes lean toward over-blocking rather than under-blocking — fail-safe > fail-open. This is the explicit OpenAI policy stance.

---

## 5. Capacity Estimates

> **Assumptions:**
> - API calls/day: 1B (ChatGPT + API)
> - Peak QPS: 200K (global; not single data center)
> - Stage 2 hit rate: 90% (most requests stop here)
> - Stage 3 hit rate: 10% of Stage 2 (borderline cases)
> - Stage 2 model size: 250MB (distilled BERT); Stage 3: 1.5GB (full BERT)
> - Human review queue: 0.005% of requests → 50K items/day
> - Log retention: 90 days hot (BigQuery), 7 years cold (compliance)

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 1B / 86,400 | **~11,600 QPS** |
| Peak QPS | 200K (stated; ~17× avg; accounts for global peak alignment) | **200K QPS** |
| Stage 2 model server QPS | 200K × 90% | **~180K QPS** |
| Stage 3 model server QPS | 200K × 10% | **~20K QPS** |
| Serving replicas (Stage 2, distilled BERT) | 180K QPS / 2,000 QPS per A10G | **~90 A10G GPUs** |
| Serving replicas (Stage 3, full BERT) | 20K QPS / 200 QPS per A10G | **~100 A10G GPUs** |
| Redis reads/s (pattern library) | 200K QPS × 1 read | **~200K reads/s** |
| Kafka ingest (decision logs) | 200K events/s × 500B/event | **~100MB/s → ~9TB/day** |
| Human review queue (daily) | 200K × 86,400 × 0.00005% | **~864K items/day** — recalculate: 200K × 86,400 × 0.005% = **~864K too high**; use 1B × 0.005% = **~50K items/day** ✓ |
| Training data (labeled, daily) | 50K human-reviewed items/day × 10KB | **~500MB/day** (human labels are expensive; volume is intentionally small) |
| Model artifact storage | 10 versions × 2GB | **~20GB** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **LLM-as-judge for borderline cases:** GPT-4 as the Stage 3 evaluator (instead of BERT) would dramatically improve nuanced edge-case accuracy but adds 500ms+ latency. Viable if moved to async human review pre-prioritization (not inline)
- [ ] **Cross-turn policy violation:** A conversation where no single turn violates policy but the cumulative interaction pattern does (slow jailbreak via multi-turn context building). Requires session-level classifier — currently only per-turn
- [ ] **Enterprise policy customization:** Enterprises want to configure their own thresholds and categories (e.g., medical provider allowing detailed drug information). Policy engine needs per-tenant configuration — scope not fully designed
- [ ] **Red team SLA:** Current target is 24hr red team → pattern library update. Is this fast enough for rapidly spreading jailbreaks on social media? Real incident suggested < 4hr is needed

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: bias parity target stated upfront (±5% FPR); false negative tolerance differentiated by category severity
- [x] Data Modeling: RLHF calibration; annotation bias mitigation; demographic proxy evaluation
- [x] ML Pipeline: SHAP explainability; production debugging path (adversarial augmentation)
- [x] Failure Modes: fail-safe stance explicit; bias drift response pathway defined
- [x] Capacity: Stage 2 vs. Stage 3 GPU counts separated; human review volume realistic

#### Recommended Follow-up Problems
- Stripe Fraud Detection — same two-stage (rules + ML) pattern; adversarial dynamics parallel
- OpenAI Scalable Recommender System — second documented OpenAI question; different domain

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) | Paper | RLHF foundation; reward model training; policy calibration |
| Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022) | Paper | Policy-based moderation via AI feedback; alternative to pure human labeling |
| OpenAI Blog: "Our approach to AI safety" | Blog | OpenAI's safety philosophy; fail-safe stance; red team cadence |
| Moderation API docs (OpenAI) | Docs | Production moderation categories and endpoints; publicly documented |
| Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP, NeurIPS 2017) | Paper | SHAP for explaining moderation decisions to human reviewers |
| Gehman et al., "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models" (2020) | Paper | Toxicity evaluation benchmark; detection methodology |
| Xu et al., "Recipes for Safety in Open-domain Chatbots" (Meta AI, 2020) | Paper | Multi-stage safety pipeline; rule + classifier cascade |
| Röttger et al., "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection" (2021) | Paper | Bias in hate speech detection; demographic FPR disaggregation |
| Bommasani et al., "On the Opportunities and Risks of Foundation Models" (Stanford CRFM, 2021) | Paper | Foundation model risks; moderation challenges at scale |
| Perez et al., "Ignore Previous Prompt: Attack Techniques for Language Models" (2022) | Paper | Prompt injection attacks; adversarial input taxonomy for rules engine |
