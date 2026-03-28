# Netflix Ads Bidding System — ML System Design

**Domain:** `ads`
**Target Company:** Netflix (Core Ads Algorithms team — MLS5 role)
**Difficulty Bar:** L6 (E6)
**Date:** 2026-03-27
**Related Designs:** `pinterest-ads-ranking.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★★ | — |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Cross-advertiser auction interference during budget exhaustion partially addressed |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Attribution for brand vs. performance ads requires different measurement frameworks — mixing them in one system degrades both. Separate ROAS measurement from CPM measurement explicitly.

---

## 1. Requirements

#### Context
Netflix launched its ad-supported tier in November 2022. Unlike Google/Meta (pure ad platforms), Netflix is simultaneously:
- **SSP** (supply-side platform): owns the ad inventory (mid-roll slots, homepage placements)
- **DSP** (demand-side platform): runs bidding on behalf of advertisers to optimize their CPA/ROAS

This creates a unique dynamic: Netflix controls both the auction and the bidder, unlike an open exchange.

#### Functional Requirements
1. Implement autobidding: given an advertiser's goal (target CPA, target ROAS, maximize conversions within budget), compute an optimal bid per ad impression opportunity
2. Run a first-price or second-price auction across eligible ads per impression slot
3. Predict pCTR and pCVR for each (ad, user, context) triple at serving time
4. Pace advertiser budgets across the day without under- or over-delivery
5. Support offline evaluation (counterfactual replay) and online A/B testing with holdout groups
6. Measure and report CPA/ROAS accurately with 30-day attribution windows

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Serving latency (p99) | ≤ 100ms | Inserted during content playback or UI render; user perception threshold |
| Availability | 99.99% | Ad serving outage = lost revenue; 1 hour outage at scale = ~$1M+ revenue loss |
| Budget delivery accuracy | ±5% of daily budget | Over-delivery violates advertiser commitment; under-delivery = lost revenue |
| pCTR calibration error (ECE) | < 2% | Miscalibrated pCTR distorts auction allocation and payment |
| Attribution window | 30 days post-click | Subscription conversions are long-cycle |
| Throughput | ~5M ad impressions/day | ~40M ad-tier MAU × ~0.1 ad opportunities per session average |

#### Scale Numbers
- **Ad-tier MAU**: ~40M (as of early 2026; growing)
- **Ad impressions/day**: ~5M (conservative; mid-roll + homepage placements)
- **Active advertisers**: ~10K (brands, studios, consumer goods)
- **Peak QPS for auction**: ~500 req/sec (primetime viewing hours in US/Europe)
- **Attribution events**: lower cardinality than ads (subscription signups, not purchases)

#### Netflix-Specific Constraints
- **Privacy**: Netflix subscribers are paying members. Privacy expectations are high. Cross-app tracking (IDFA/GAID) is restricted. Rely on Netflix first-party data for targeting.
- **Content-adjacency**: ads appear near content the member chose. A horror ad before a comedy or a children's title is a brand safety violation — for both the advertiser and Netflix.
- **Subscriber experience first**: guardrail: ad load cannot increase DAU churn. Every bidding experiment includes subscriber churn as a guardrail metric.

#### Out of Scope
- Ad creative generation
- Targeting segment definition (owned by separate data/ML team)
- Content recommendation (separate system; this system is ads-only)

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| User content history (genres, completion rate, search) | `batch` | daily | Netflix content system → feature store | Rich first-party signal; privacy-safe (no cross-app) |
| User account features (plan type, tenure, payment method country) | `batch` | daily | Account system → feature store | Subscription tier signals purchase intent |
| Current viewing context (title genre, maturity rating, episode position) | `real-time` | request-time | Playback session → request | Content adjacency for brand safety |
| Ad creative features (brand, category, sentiment, visual embeddings) | `batch` | per creative upload | Ad creative store | Used for brand safety matching + pCTR |
| User-ad historical interactions (impressions, clicks per advertiser category) | `batch` | daily | Interaction logs → feature store | Frequency + relevance signal |
| Time features (hour, day_of_week, is_weekend) | `real-time` | request-time | Request context | Viewership patterns shift primetime |
| Advertiser budget remaining | `real-time` | ≤ 5 min | Budget tracker → Redis | Pacing input |
| Current surge/load (ad request rate) | `real-time` | ≤ 60s | Ad server metrics | Traffic shaping signal |

#### Label Definition

**pCTR label:**
- Binary: did the user click the ad? Positive rate ~1–3% (typical display click rate)
- Label delay: immediate (click is observed within seconds)
- Position bias: ads shown in position 1 of a mid-roll block get more clicks — apply position debiasing (IPS or position as feature)

**pCVR label:**
- Binary: did the user subscribe (or take target action) within the attribution window?
- Positive rate: ~0.1–0.5% (subscription is high-intent but low-frequency)
- Label delay: 30 days — critical issue; must apply truncated attribution correction
- Multi-touch attribution: a user may see 5 Netflix ads before subscribing. Last-touch overweights the final ad; data-driven attribution (Shapley values) distributes credit across touchpoints.

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| User feature store (batch) | Feast → Redis (online) + S3 Parquet (offline) | Sub-ms online reads; S3 for training |
| Ad creative store | Redis (creative_id → embedding + metadata) | Low cardinality (~1M creatives); fast lookup |
| Budget tracker | Redis (advertiser_id → remaining_budget, TTL: daily) | High-write (decremented per win); atomic operations |
| Auction logs | Kafka → S3 Parquet (partitioned by date, advertiser) | Training data + counterfactual analysis |
| Attribution events (conversions) | Kafka → Hive (partitioned by click_timestamp + attribution_date) | Long-window joins; Spark-based |
| A/B assignment store | Redis (user_id → experiment_arm) | Per-request lookup; stable assignment per user |

---

## 3. ML Pipeline

#### System Architecture

```
[User starts stream / opens app]
         ↓
Ad Opportunity Detected (mid-roll trigger or homepage slot)
         ↓
Ad Request → Ad Server (p99 < 100ms budget)
         ↓
┌─────────────────────────────────────────────────────┐
│                  Auction Pipeline                   │
│                                                     │
│  1. Candidate Retrieval                             │
│     Brand-safety filter: exclude mis-matched ads    │
│     Frequency cap: exclude ads over impression cap  │
│     Budget filter: exclude exhausted advertisers    │
│     → ~50 candidate ads (from ~10K total)           │
│                                                     │
│  2. Scoring (pCTR × pCVR)                          │
│     Model: GBDT + LR stack (calibrated)             │
│     Input: user features + ad features + context    │
│     Output: pCTR, pCVR per candidate                │
│                                                     │
│  3. Bid Calculation (autobidding)                   │
│     bid_i = pCVR_i × conversion_value / λ           │
│     λ = Lagrangian multiplier (updated by CPA loop) │
│                                                     │
│  4. Auction (second-price GSP)                      │
│     Winner = argmax(bid_i × quality_score_i)        │
│     Payment = next_effective_bid / own_quality + ε  │
│                                                     │
│  5. Pacing check                                    │
│     Apply pacing multiplier ρ ∈ [0,1]               │
│     ρ < 1 if ahead of spend pace                    │
└─────────────────────────────────────────────────────┘
         ↓
Ad Served → Impression Logged → Click/Conversion Tracked
```

#### pCTR / pCVR Models

**Architecture**: GBDT (LightGBM) + Logistic Regression stack
- GBDT learns non-linear feature interactions (user genre × ad category × time-of-day)
- LR on GBDT leaf encodings: fast calibrated probability output
- Why not deep learning for initial build: Netflix's ad inventory is smaller than Meta/Google. GBDT + LR is faster to iterate, more interpretable, and easier to calibrate. Move to DLRM/two-tower when scale justifies it.

**Calibration**: mandatory post-hoc calibration with Platt scaling or isotonic regression. ECE < 2% target. Recalibrate after every model refresh.

**Training data**:
- pCTR: all impressions (with position as feature) from last 30 days
- pCVR: clicks from last 90 days with 30-day attribution window closed (no truncated labels)
- Class imbalance: pCVR positives are ~0.1% → use negative downsampling (10:1) with inverse probability reweighting to restore calibration

#### Autobidding (CPA/ROAS Control)

**Per-advertiser λ control loop (outer loop, 1-hour cadence):**
```
observed_CPA = total_spend / attributed_conversions (lag-corrected)
error = observed_CPA - target_CPA
λ_new = λ_old + Kp × error + Ki × integral(error)
bid_i = pCVR_i × conversion_value / λ_new
```
- Attribution lag correction: conversions observed so far / expected completions at this hour of day (based on historical attribution completion curve)
- Anti-windup: clamp λ to [target_CPA × 0.5, target_CPA × 2.0] — prevents wild oscillation on new campaigns

**Budget pacing (inner loop, 5-min cadence):**
```
pacing_factor ρ = remaining_budget / E[remaining_spend_to_end_of_day]
effective_bid = bid_i × ρ
```
- `E[remaining_spend]` from a spend forecast model (LightGBM on historical spend patterns by advertiser × time-of-day × day-of-week)
- ρ > 1.0 capped at 1.0 (never overbid just because behind pace)

#### Attribution Pipeline

```
[Click Event] → Kafka → Attribution Service
                              ↓ (joins within 30-day window)
[Conversion Event] → Kafka → Attribution Service
                              ↓
                    Attribution record: (click_id, conversion_id, value, lag_days)
                              ↓
                    Hive (partitioned by click_date)
                              ↓
                    Daily Spark job → CPA/ROAS metrics per advertiser per campaign
```
- Multi-touch: data-driven attribution using Shapley values across the click path
- Deduplication: same user, same conversion, multiple ad clicks → attribute fractionally
- Privacy-safe join: all joins happen within Netflix's own data (no third-party cookie)

---

## 4. Failure Modes

| Failure | Detection | Mitigation |
|---|---|---|
| pCTR model serving down | Model health check; latency spike alert | Fall back to category-average CTR per (ad_category, user_segment) |
| Budget tracker unavailable | Redis latency alert > 10ms | Throttle all bids to 50% of normal rate; over-delivery < 10% acceptable short-term |
| Attribution pipeline lag > 24h | Conversion count drop alert | Suppress CPA-dependent λ updates; hold λ at last known value |
| Advertiser over-delivery | Daily spend > target + 10% | Hard spend cap enforced at budget tracker; refund excess to advertiser |
| pCTR calibration drift (ECE > 5%) | Daily calibration check | Trigger emergency recalibration job; pause model updates until resolved |
| Brand safety violation (wrong ad with wrong content) | Post-hoc audit on content × ad category logs | Pre-auction brand safety filter is P0 — failure here is a business/PR incident |
| Auction revenue collapse (most advertisers budget-exhausted by 6pm) | RPM drop alert | Redistribute budget delivery earlier; alert account managers to increase budgets |
| A/B experiment contamination (holdout users exposed) | User assignment audit | Holdout is a hard exclude in the assignment service; any exposure is a bug, not acceptable |

---

## 5. Capacity Estimates

| Component | Estimate | Assumptions |
|---|---|---|
| Ad auction QPS | 500 req/sec peak | 5M impressions/day ÷ 86,400 × 10× peak/average |
| Candidates per auction | ~50 | After brand safety + frequency + budget filters from ~10K advertisers |
| pCTR/pCVR inference | 25,000 scores/sec | 500 QPS × 50 candidates |
| Feature store reads | ~100K/sec | 500 QPS × ~200 features, batched per request |
| Attribution joins | ~5M/day | 5M impressions × ~1% click rate × join to conversion stream |
| Budget tracker writes | ~500/sec | One write per auction win |
| Auction log storage | ~2 TB/month | 5M impressions/day × 1 KB/record × 30 |

---

## 6. Evaluation Framework (Netflix-Specific)

#### Offline Evaluation
- **Counterfactual replay**: take historical auction logs; apply new bidding algorithm's bids; estimate outcomes via doubly robust off-policy evaluation
- **Calibration tests**: ECE per (advertiser category, user segment, time-of-day) bucket — calibration degrades on rare segments first
- **Budget delivery simulation**: simulate spend trajectory under new pacing algorithm against historical traffic

#### Online Evaluation (A/B + Holdout)
- **Randomization unit**: advertiser (not user) — bidding changes affect advertiser outcomes, not individual impressions
- **Primary metrics**: CPA (cost per attributed subscription), ROAS, budget delivery rate
- **Guardrail metrics**: subscriber churn rate (P0), content engagement rate (engagement must not fall due to ad experience), pCTR calibration ECE
- **Long-term holdout**: 2% of advertisers permanently held out from new bidding algorithm. Measure at 30/90/180 days for ROAS convergence and advertiser retention.
- **Novelty effect concern**: new bidding algorithms often over-perform early (finding easy wins in unexploited auction dynamics) then regress. Runtime minimum: 4 weeks. Measure effect in weeks 3–4 separately from weeks 1–2.

#### Attribution Measurement
- 30-day attribution window with completion correction
- Incrementality testing: geo-based holdout (no ads in selected DMAs) to estimate true incremental lift vs. organic subscription rate
- Advertiser-reported ROAS vs. Netflix-measured ROAS: reconciliation process. Discrepancies > 20% trigger audit.

---

## 7. Netflix-Specific Design Decisions

**Why second-price auction (GSP) vs. first-price?**
Netflix's ad platform is new. Advertisers bidding in a first-price auction need bid shading sophistication that most of Netflix's initial advertiser base (brand advertisers) doesn't have. GSP is simpler for advertisers, reduces barrier to entry, and preserves revenue equivalence in equilibrium. Move to first-price as the advertiser base matures.

**Why GBDT+LR vs. deep learning for pCTR/pCVR?**
At ~5M impressions/day, Netflix's ad volume is 1,000× smaller than Meta/Google. GBDT+LR trains in hours (not days), is interpretable for debugging, and calibrates reliably with Platt scaling. The step to DLRM/two-tower architecture is justified when volume exceeds ~100M impressions/day and feature interactions become too complex for GBDT.

**Why first-party data only?**
iOS App Tracking Transparency (ATT) and Chrome's deprecation of third-party cookies make cross-app tracking unreliable. Netflix's first-party viewing data (genre preferences, completion rates, search history) is both richer and more stable than third-party data. This is a long-term moat — competitors rely on cookies Netflix doesn't need.

**Why keep brand safety as a pre-auction filter (not a ranking signal)?**
A horror ad adjacent to a children's title is a binary violation — there's no "slightly wrong" brand safety outcome. Making brand safety a ranking signal means it could be outweighed by bid value. Hard pre-filter ensures it's a constraint, not a trade-off.
