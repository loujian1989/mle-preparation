# Ads & Auction Theory — ML Knowledge Q&A

P0: Reddit (Staff), Pinterest, Meta. Critical for any ads-adjacent MLE role.

---

## Auction Mechanics

### Q: What is the difference between VCG and GSP auctions? Which does Google/Meta use and why?

**Answer (Staff level):**
- **VCG (Vickrey-Clarke-Groves)**: theoretically truthful — each bidder pays the externality they impose on others (i.e., the opportunity cost to other bidders). Dominant strategy: bid true value. Maximizes social welfare.
- **GSP (Generalized Second Price)**: practical approximation. Each winner pays the next-highest bid (or quality-score-adjusted next bid). NOT fully truthful (bidders can shade bids strategically), but simple to compute.
- **Google/Meta use a quality-score-weighted GSP**: `effective_bid = bid × quality_score`. This means a high-quality ad can outrank a higher-dollar bid. The payment formula is: `payment = (next_effective_bid / own_quality_score) + ε`.
- **Why GSP over VCG?**: VCG is computationally complex at scale (must compute the optimal allocation without each bidder), vulnerable to collusion, and revenue is harder to predict. GSP is simpler, well-understood by advertisers, and empirically near-optimal.
- **Revenue note**: In equilibrium, GSP under-collects vs. VCG (Nash equilibrium bids in GSP are not equal to true values). But in practice, advertiser behavior in both auctions converges.

**Company context:** Reddit (Staff ads auction role — this is the core interview question), Pinterest (ads ranking), Meta.

**Common wrong answer:** "Google uses a second-price auction." — Incomplete. The quality score weighting is the critical detail. Without it you get irrelevant ads in top positions. Also: confusing "second-price" with VCG (they differ when there are multiple positions).

---

## pCTR and Quality Score

### Q: What is a quality score, why does it exist, and how do you model pCTR?

**Answer (Staff level):**
- **Quality score** (Google terminology) or **relevance score** (Meta): a measure of ad quality that combines pCTR, ad landing page relevance, and expected user experience. Purpose: prevent auction gaming by high-budget/low-quality ads.
- Without quality score: auction becomes "highest bidder wins" → irrelevant ads in top slots → user experience degrades → platform trust declines.
- **Effective rank = bid × pCTR** (simplified). A $1 bid with pCTR=10% outranks a $2 bid with pCTR=3%.
- **pCTR modeling**:
  - Features: ad creative features (text, image embedding), user features (demographics, browsing history), context features (query, page, placement), historical CTR (at ad / advertiser / category level).
  - Model: GBDT + LR stacking is historically dominant (fast inference, calibrated). DLRM at Meta scale.
  - **Calibration is critical**: pCTR is used directly in the auction formula, so miscalibrated probabilities distort allocation. ECE < 2% is a common production requirement.
  - **Training data bias**: ads shown in position 1 have higher observed CTR — must use position as a feature or apply position debiasing (similar to IPS in organic ranking).

**Company context:** Reddit (ads auction), Pinterest, Meta. Reddit Staff role specifically requires designing the full pCTR pipeline.

**Common wrong answer:** "pCTR is just the historical CTR of the ad." — Historical CTR is one feature but suffers from sparse data for new ads and position confounding. pCTR is a model output, not an average.

---

## Revenue vs. Relevance

### Q: How do you balance revenue maximization with user experience in ads?

**Answer (Staff level):**
- Pure revenue optimization (maximize expected revenue = bid × pCTR) ignores user experience. A click-bait ad maximizes short-term revenue but degrades retention → long-term revenue falls.
- **Multi-objective auction**: augment effective rank with a relevance/quality term:
  `rank = α × bid × pCTR + β × relevance_score`
  where `β` is tuned by A/B experiments measuring revenue vs. session-depth / return-rate trade-off.
- **Revenue-per-query cap**: if a query generates too many ads, organic results are suppressed → user satisfaction drops. Cap ads per page.
- **North star metric disagreement**: ads team optimizes Revenue Per Mille (RPM); product team optimizes DAU/retention. Staff-level answer acknowledges the tension and proposes joint metric governance (e.g., guardrail constraints: "revenue can increase but organic engagement cannot fall by more than X%").

**Company context:** Reddit (Staff), Pinterest, Meta. This is an organizational/product-ML intersection question common at Staff bar.

**Common wrong answer:** "I'd maximize expected revenue = pCTR × bid." — No user experience guard. Staff answer explicitly names the long-term revenue / short-term revenue trade-off and proposes constraints.

---

## Calibration in Ads

### Q: Why is calibration especially critical in auction-based ad systems?

**Answer (Staff level):**
- In organic ranking, miscalibration affects rank order (bad) but not the explicit dollar value. In auctions, pCTR appears directly in the revenue formula: `expected_revenue = bid × pCTR`. Miscalibration distorts expected revenue, leading to:
  1. **Wrong allocation**: a 2× overestimated pCTR inflates rank and the advertiser "wins" unfairly, displacing a higher-quality competitor.
  2. **Wrong payments**: payment = next_bid / own_quality_score. Miscalibrated quality score → incorrect charge to advertiser.
  3. **Budget mispacing**: advertiser bids are calibrated to ROI assumptions; overestimated CTR means advertiser overpays, budget depletes early → advertiser churn.
- **Production requirement**: calibrate pCTR models with isotonic regression or Platt scaling after every major model refresh. Monitor calibration lift across ad categories (calibration often degrades on rare categories or new ad formats).

**Company context:** Reddit (Staff auction MLE), Pinterest, Meta ads.

**Common wrong answer:** Treating calibration as a minor detail. At Staff bar, calibration in ads is a P0 reliability concern, not an afterthought.

---

## Bid Shading and Auctions Moving to First-Price

### Q: Google Display Network moved from second-price to first-price auctions. What changed for advertisers and for the ML system?

**Answer (Staff level):**
- **Second-price**: advertiser pays next-bid + ε. Dominant strategy = bid true value. Simple for advertiser, good for revenue predictability.
- **First-price**: advertiser pays their own bid. Dominant strategy = shade bid below true value (bid lower than value to preserve margin). Bidder must estimate optimal shade factor = `bid = value × (1 − 1/k)` (Vickrey shading for k bidders, simplified).
- **What changed for advertisers**: must model optimal bid (bid shading algorithms, often ML-based using historical win rates). Advertisers who don't shade overpay → churn.
- **What changed for the ML system**:
  - pCTR model must account for strategic bidding behavior — bids are no longer truthful.
  - Win probability estimation: model must predict P(win | bid) to help advertisers set shaded bids.
  - Revenue prediction is harder: win rate × payment is no longer second-bid-determined.
- **Trend**: most large auctions (Google, AppNexus) shifted to first-price 2019–2021. Header bidding ecosystems are first-price.

**Company context:** Reddit Staff (ads infrastructure deep-dive), Pinterest.

**Common wrong answer:** "First-price is better because the platform earns more." — Not necessarily. With optimal shading, equilibrium revenue is equivalent (Revenue Equivalence Theorem). The ML complexity cost is the key differentiator.

---

## Frequency Capping

### Q: How do you implement frequency capping, and what's the ML implication?

**Answer (Staff level):**
- **Frequency cap**: limit impressions of a given ad creative to a user within a time window (e.g., max 5 impressions/day/creative). Goal: prevent ad fatigue (click rate collapses after N impressions of same ad), protect user experience.
- **Implementation**: distributed counter per (user_id, ad_id, window) in a fast store (Redis or custom). At request time: `if count >= cap: exclude ad from auction candidates`. Cap check happens at retrieval stage, before ranking.
- **ML implication**: training data includes frequency-capped ads (not shown after cap). If you train a CTR model without controlling for frequency, you learn "ad shown for 5th time has lower CTR" as a feature interaction. This is valid signal. BUT: exposure distribution at training time (user saw 3.2 impressions on average) ≠ serving time (user may see 1 impression of a new ad). Model must handle this.
- **Recency feature**: log(impressions_in_last_24h) as a model feature to explicitly capture saturation effect.

**Company context:** Reddit, Pinterest, Meta.

**Common wrong answer:** "I'd filter frequency-capped ads before the auction." — Correct at inference time, but misses the training data implication: frequency affects CTR, so the training distribution is capped-aware. Must address the feature.

---

## Auction Mechanics (Deep Dive)

### GSP Payment Formula — Concrete Example

Three advertisers bidding for two ad slots. Slots have click-through rates (CTR) of 0.10 (slot 1) and 0.06 (slot 2):

```
Advertiser A: bid=$5, quality_score=0.8 → effective_bid = 5×0.8 = 4.0
Advertiser B: bid=$4, quality_score=1.0 → effective_bid = 4×1.0 = 4.0
Advertiser C: bid=$3, quality_score=0.9 → effective_bid = 3×0.9 = 2.7

Rank by effective_bid: A=4.0, B=4.0 (tie, A wins by timestamp), C=2.7
Slot 1 → A, Slot 2 → B

Payment for A (slot 1):
  Must pay enough that A's effective_bid just beats B's:
  payment_A = (B's effective_bid / A's quality_score) + ε
             = (4.0 / 0.8) + ε = $5.00 + ε

Payment for B (slot 2):
  Must beat C's effective_bid:
  payment_B = (C's effective_bid / B's quality_score) + ε
             = (2.7 / 1.0) + ε = $2.70 + ε
```

High quality score → lower payment (same effective bid, but divided by larger quality score).

### Why Quality Score Benefits Both Platform and Users

Without quality score: A ($5 bid, low quality) beats B ($4 bid, high quality). Users see a low-quality ad → lower click rate → platform earns `5 × low_CTR`. With quality score: B (effective 4.0) can beat A (effective 4.0 if A's quality is 0.8 and bid is 5). Platform earns `4.0 × high_CTR` per click from B — potentially more revenue AND better user experience.

### VCG vs GSP Revenue — Revenue Equivalence Theorem

In theory, under risk-neutral bidders with independent private values, both VCG and GSP yield the same expected revenue (Revenue Equivalence Theorem). In practice, GSP generates non-truthful bidding → bid shading → platform uncertainty about true values. VCG generates truthful bids but is computationally harder. Both converge empirically.

---

## pCTR and Quality Score (Deep Dive)

### Why Historical CTR Alone Fails

A new ad with zero history has no historical CTR → it can't compete. But a new, highly relevant ad should be shown.

An ad that was shown 1,000 times in high-traffic, high-CTR positions will have inflated historical CTR vs. the same ad shown in low-traffic positions.

Position confounding: an ad shown 90% at position 1 will have 3–5× higher CTR than the same ad shown at position 5, purely due to position bias.

**pCTR model input hierarchy:**
```
Level 1 (ad-specific):    creative embedding, landing page quality, format
Level 2 (advertiser):     category CTR, account history, industry benchmarks
Level 3 (user):           browsing history, demographics, intent signals
Level 4 (context):        query, placement, device, time of day
Level 5 (interaction):    user × ad affinity, user × category history
```

The model combines all these to produce a calibrated probability. New ad with no history → falls back to advertiser and category priors.

### Calibration Requirement — Why ±2% ECE Matters

In organic ranking: a miscalibrated model just reorders items (sub-optimal but not catastrophic).

In auction: `expected_revenue = bid × pCTR`. If pCTR is 2× too high:

```
True pCTR = 0.03, advertiser bid = $10
Correct auction payment = $10 × 0.03 = $0.30 expected revenue per impression
Miscalibrated pCTR = 0.06: effective bid = $10 × 0.06 = $0.60
→ this ad beats out a competitor with true effective bid $0.45
→ wrong ad wins, both revenue AND relevance are wrong
→ advertiser with pCTR=0.06 (inflated) systematically over-delivers → overspends → churns
```

---

## Bid Shading in First-Price Auctions (Deep Dive)

### Why First-Price Forces Bid Shading

In second-price: bid your true value V. You pay second-highest bid (< V). No reason to shade.

In first-price: bid your true value V. You win but pay V → zero surplus. Shade to V − ε to capture some surplus. But every advertiser does this → equilibrium bid = `V × (N-1)/N` for N symmetric bidders (Vickrey shading formula).

```
2 advertisers: equilibrium bid = V × 1/2 = 0.5V
5 advertisers: equilibrium bid = V × 4/5 = 0.8V
100 advertisers: equilibrium bid = V × 99/100 ≈ 0.99V
```

As competition increases, shading decreases (more competition → less margin to shade). In a competitive market with many advertisers, first-price and second-price equilibria converge.

### ML for Win Probability Estimation

In first-price, advertisers must model `P(win | bid, context)` to compute the optimal shade:

```
optimal_bid = argmax_b [P(win | b) × (V - b)]

P(win | b) ≈ logistic regression or GBT on:
  - Historical win rate at various bid levels in similar contexts
  - Competitor bid density (estimated from win/loss data)
  - Time of day, audience competition level

Optimal shade factor = b* / V = solve for argmax
```

This is the ML problem that autobidding systems solve: estimating the value-adjusted optimal bid under first-price dynamics.

---

## Frequency Capping (Deep Dive)

### The CTR Saturation Curve

Empirically, CTR decays after repeated exposures to the same ad:

```
Exposure 1:   CTR = 0.045  (novel, high attention)
Exposure 2:   CTR = 0.038  (still fresh)
Exposure 3:   CTR = 0.030  (familiarity declining)
Exposure 5:   CTR = 0.018  (saturation beginning)
Exposure 10:  CTR = 0.009  (strong ad fatigue)
Exposure 15+: CTR = 0.004  (near-zero marginal value)
```

The frequency cap is set where the marginal value of one more impression approaches zero. For most ad formats: cap = 3–7 per day, 10–20 per week.

### Impact on pCTR Training

Training data contains users who have seen the same ad 0, 1, 2, 3... times. The pCTR model learns this decay:

```
pCTR(user, ad, frequency=1) = 0.045
pCTR(user, ad, frequency=5) = 0.018
```

At serving time: new users (frequency=0) will be shown this ad for the first time. The model must generalize from observed frequencies to zero — which it can if `frequency` is explicitly included as a feature.

If frequency is NOT included as a feature: the model averages over all exposure counts in training data → underestimates CTR for new exposures, overestimates for high-frequency exposures → wrong auction allocation.
