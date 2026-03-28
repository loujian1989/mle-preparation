# ML Knowledge & Domain Knowledge Interview Prep

Staff/L6 bar. Q&A cards only — no tutorials. Each card has a Staff-level answer
and a "Common wrong answer" that distinguishes mid-level from Staff.

---

## Company × Topic Matrix

Use this to prioritize by your next interview.

| Topic | Meta | OpenAI | Stripe | Reddit | Netflix | Uber | Shopify | Pinterest | Roblox | Whatnot |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Evaluation Metrics** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Optimization & Training** | ✓ | ✓ | | | ✓ | | | | ✓ | |
| **Regularization** | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ | | |
| **Feature Engineering** | | | ✓ | ✓ | | ✓ | ✓ | | | |
| **Transformers & Attention** | ✓ | ✓ | | | | | | | | |
| **Training at Scale** | ✓ | ✓ | | | | | | | | |
| **RLHF & Alignment** | | ✓ | | | | | | | | |
| **Tree Models & GBM** | | | ✓ | ✓ | ✓ | | ✓ | | | |
| **Embeddings & Retrieval** | ✓ | | | | ✓ | | | ✓ | ✓ | |
| **Ranking & Recommendation** | ✓ | | | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| **Ads & Auction Theory** | ✓ | | | ✓ | | | | ✓ | | |
| **Autobidding & Pacing** | ✓ | | | ✓ | | | | ✓ | | |
| **Dynamic Pricing & Mechanism Design** | | | | | | ✓✓ | | | | |
| **Fraud & Trust Safety** | | ✓ | ✓ | | | | | | | |
| **Real-Time ML** | | | | | | ✓ | ✓ | | ✓ | ✓ |

---

## Directory Structure

```
docs/ml-knowledge/
├── README.md                            ← this file
├── core-ml/
│   ├── evaluation-metrics.md            ← P0: universal
│   ├── optimization-and-training.md     ← P0: OpenAI, Meta, Netflix
│   ├── regularization.md                ← P0: universal
│   └── feature-engineering.md          ← P1: Stripe, Shopify, Uber
├── deep-learning/
│   ├── transformers-and-attention.md    ← P0: OpenAI, Meta
│   ├── training-at-scale.md             ← P1: OpenAI, Meta
│   └── rlhf-and-alignment.md            ← P0 (OpenAI only)
├── classical-ml/
│   ├── tree-models-and-gbm.md           ← P1: Stripe, Reddit, Shopify
│   └── embeddings-and-retrieval.md      ← P1: Meta, Pinterest, Roblox
└── domain/
    ├── ranking-and-recommendation.md    ← P0: Reddit, Netflix, Meta, Pinterest, Roblox
    ├── ads-and-auction-theory.md        ← P0: Reddit (Staff), Pinterest, Meta
    ├── autobidding-and-pacing.md        ← P0: Reddit (Staff), Pinterest, Meta
    ├── dynamic-pricing-and-mechanism-design.md ← P0: Uber Supply Pricing team
    ├── fraud-and-trust-safety.md        ← P0: Stripe, OpenAI
    └── real-time-ml.md                  ← P1: Uber, Shopify, Roblox, Whatnot
```

---

## Interview Format by Company

| Company | Round Name | Format | Key Probe |
|---|---|---|---|
| **OpenAI** | Project deep-dive | 60 min, defend past work | Training dynamics, RLHF, reward hacking |
| **Meta** | ML design (conceptual) | Within 45-min design round | DLRM, embeddings, feature engineering at scale |
| **Stripe** | ML phone screen | 30 min before take-home | Evaluation metrics, imbalance, feature leakage |
| **Reddit** | Live model building | 45 min with live dataset | Ranking domain, IPS, auction mechanics |
| **Netflix** | Take-home quiz | Async (24h) | Product-tied metrics, A/B holdout design |
| **Uber** | Onsite ML depth | 45 min | Geospatial features, prediction intervals |
| **Shopify** | ML knowledge screen | 30 min | Calibration, leakage, SOLID |
| **Pinterest** | ML design round | 45 min | Ads ranking, spam detection, two-tower |
| **Roblox** | Onsite rounds | 45 min | Recommendation cold start, platform |
| **Whatnot** | ML design hybrid | 45 min | Real-time ranking, fairness |

---

## How to Use This Section

1. **Before each interview**: open the README, find the company row, open every ✓ file
2. **Inside each file**: read the Q, cover the answer, say it aloud, then compare
3. **Focus on "Common wrong answer"**: this is where the bar differentiates
4. **For OpenAI**: `rlhf-and-alignment.md` is non-negotiable; add `transformers-and-attention.md`
5. **For Reddit Staff**: `ads-and-auction-theory.md` + `autobidding-and-pacing.md` + `ranking-and-recommendation.md` are the core
