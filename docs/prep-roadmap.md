# MLE Interview Prep: 10-Week Strategy for Staff/Senior Roles
**Target Companies:** TTD, Stripe, Uber, Meta, OpenAI, Shopify, Netflix, Roblox
**Background:** PhD CS (game theory/mechanism design), DoorDash Ads (auction/pricing/autobidding), Amazon (Sponsored Products)

---

## 1. The Overlap Strategy

### Coding — Universal Core (applies to all 8)

| Topic | Depth Required | Priority |
|---|---|---|
| Graphs (BFS/DFS, shortest path, topological sort) | Medium-High | P0 |
| Dynamic Programming (1D/2D, interval, knapsack) | High | P0 |
| Two Pointers / Sliding Window | Medium | P0 |
| Heaps / Priority Queues | High (real-time systems) | P0 |
| Hash Maps / Set manipulation | Medium | P0 |
| Binary Search (on value, not just index) | Medium | P1 |
| Trees (LCA, serialization, path problems) | Medium | P1 |
| System design coding (rate limiters, LRU cache, task scheduler) | High | P0 |

**Non-negotiables across all 8:**
- Clean code, typed signatures, edge case narration before coding
- Complexity analysis stated before optimization pass
- Think aloud — all 8 companies score communication as a dimension

### ML System Design — Universal Core

| Pillar | What Every Company Expects |
|---|---|
| Problem framing | Translate vague product goal → ML objective → metric (online + offline) |
| Data pipeline | Feature store, training data construction, label generation, skew |
| Model selection | Justify architecture vs. baseline; know when not to use deep learning |
| Training infra | Distributed training, checkpointing, experiment tracking |
| Serving | Online vs. batch, latency SLA, caching, model versioning |
| Monitoring | Drift detection, shadow scoring, data quality alerts, feedback loops |
| Failure modes | Fallbacks, circuit breakers, degraded-mode behavior |

**Core system designs to master (used by 6+ of 8 companies):**
- Two-tower retrieval + re-ranking pipeline (Meta, Netflix, Roblox, Uber)
- Ads auction + bidding system end-to-end (TTD, Meta, Uber)
- Real-time feature serving with low-latency SLA (all 8)
- A/B testing + experimentation platform design (all 8)

### Behavioral — Universal Core (STAR, Staff-bar calibrated)

**3 anchor stories you must have polished:**
1. **Ambiguity → Clarity:** Led an underspecified project from mess to shipped — quantify impact
2. **Cross-functional Influence:** Changed technical direction of a team/org without authority
3. **Hard Trade-off:** Made a decision where the "correct" answer was genuinely unclear

**Axes every company scores:**
- Scope of impact (team → org → company)
- Dealing with ambiguity
- Technical judgment under constraints
- Leveling up others

---

## 2. The Ads Economics & Mechanism Design Advantage

Your background is a genuine moat for ~5 of the 8 companies. Here's how to deploy it surgically.

### Narrative Angles by Company

**The Trade Desk (TTD) — Programmatic Auction Design**
- TTD is a DSP. Their core problem is real-time bidding: win the right impression at the right price at <100ms.
- Your angle: "At DoorDash, I designed autobidding systems where the advertiser states a goal (CPA/ROAS) and the system bids optimally on their behalf — this is the same optimization objective as TTD's Koa AI. The challenge is budget pacing under uncertainty with non-stationary supply."
- Mechanism design hook: bring up bid shading (first-price auction strategies), pacing as a Lagrangian relaxation of a constrained optimization, and the trade-off between exploration and exploitation in budget allocation.
- Key TTD-specific angle: campaign delivery guarantees (programmatic guaranteed vs. open auction), data marketplace privacy (UID2.0).

**Meta — Ads Auction at 1B+ Scale**
- Meta runs a generalized second-price variant with quality scores baked in (eCPM = bid × quality).
- Your angle: "My work at Amazon and DoorDash maps directly — specifically the interaction between auction revenue, advertiser value maximization, and long-term platform health (a multi-objective problem)."
- Mechanism design hook: social welfare vs. revenue maximization trade-off, incentive compatibility, why GSP is not truthful and what that means for autobidder convergence.
- Differentiation: most candidates explain the auction; you should explain why it's designed that way (VCG → GSP approximation, strategyproofness relaxation for computational tractability).

**Uber — Marketplace Pricing & Matching**
- Uber's core ML problems: ETA, surge pricing, driver-rider matching.
- Your angle: "Marketplace mechanism design is what connects our backgrounds — DoorDash dispatch and Dasher incentive design is structurally identical to Uber's driver supply/demand balancing. Both involve online matching under uncertainty with strategic agents."
- Mechanism design hook: matching markets, stable matching, VCG-based incentive mechanisms for drivers, surge as a price-discovery mechanism not just demand suppression.
- Concrete: design a surge pricing system from scratch and narrate the mechanism design choice (posted price vs. auction vs. ML-predicted price) as a first-principles decision.

**Stripe — Payments, Fraud, Risk Pricing**
- Not a classic auction company, but mechanism design applies to: risk-based pricing (setting interchange-equivalent fees), fraud detection (adversarial ML as a game between fraudster and detector).
- Your angle: "Fraud detection is a Stackelberg game — the defender commits to a policy, the attacker best-responds. My mechanism design background gives me a formal framework for thinking about equilibria in adversarial settings."
- Concrete: design a fraud scoring system and explicitly discuss the adversarial feedback loop, how publishing model scores enables gaming, and why some features must be withheld.

**OpenAI — Alignment, RLHF, Multi-agent**
- Your game theory background has direct application: multi-agent RL, Nash equilibria in RLHF reward model training, mechanism design for alignment.
- Angle: "Mechanism design asks: given strategic agents with private preferences, can you design an incentive structure that elicits truthful behavior? RLHF is doing the same thing — designing a reward signal so that the model 'reports' aligned behavior."

### The 30-Second Pitch Template (use in intros)
> "I'm a Senior MLE at DoorDash focused on ads economics — specifically auction mechanism design, autobidding, and budget pacing. My PhD is in computational game theory, so I approach marketplace ML problems by first modeling the strategic behavior of agents — advertisers, platforms, users — before choosing an ML approach. That lens is particularly useful for systems where the model's outputs affect future inputs, which is true in any closed-loop bidding or pricing system."

---

## 3. Company-Specific Deep Dives (The Deltas)

### Cluster A: Ads/Marketplace ML — TTD, Meta, Uber
**Common thread:** Real-time bidding, two-sided marketplace, high-throughput feature serving
**Your delta:** Minimal — this is your core domain
**What to add:**
- TTD: Programmatic ecosystem literacy (DSP/SSP/DMP/ad exchange topology), UID2.0 privacy-preserving identity
- Meta: DLRM architecture, FAISS/ScaNN for embedding retrieval, Flink feature pipelines
- Uber: H3 geospatial indexing, Michelangelo feature store, prediction intervals for ETA (not just point estimates)

### Cluster B: Practical Engineering — Stripe, Shopify
**Common thread:** Production code quality, system design from first principles, debugging, API design
**Your delta:** Significant — less about ML, more about software craft
**Preparation:**
- Stripe: Expect a take-home or live coding exercise on real-world data (not LeetCode). Practice building a complete ML pipeline (ingest → features → model → API) in clean Python. Know Pydantic, gRPC schemas, and idempotency.
- Shopify: SOLID principles enforced at class level. Every design decision requires a decision log. Prepare ≥2 alternatives for any system design. Rails/Ruby is less relevant for MLE roles; focus on clean Python service architecture.
- Both: Expect debugging exercises. Practice reading broken code, identifying the failure mode, and explaining it systematically.

**Must-practice for this cluster:**
- Design a webhook delivery system (idempotency, retries, ordering guarantees)
- Implement a rate limiter (token bucket vs. sliding window — know the trade-off)
- Build an ML inference API (request validation, batching, timeout handling, circuit breaker)

### Cluster C: Personalization + Emerging Ads ML — Netflix, Roblox
**Common thread:** Recommendation systems, user modeling, A/B experimentation
**Your delta:** Lower than originally estimated — both companies are actively building ads ML stacks (validated March 2026)

#### Netflix
**Ads context (verified):** Netflix is actively hiring MLEs specifically for ads — open roles include "MLE L5 - Ads" (ad ranking, pacing, personalization) and "MLE L5 Senior - Ads Inventory Management & Forecasting". They are building an in-house ad tech ecosystem from scratch, making this a greenfield problem structurally similar to early-stage ads infrastructure at DoorDash/Amazon.

**Your angle:** "Netflix Ads is at the stage DoorDash Ads was 2–3 years ago — building the core auction, pacing, and forecasting systems. My experience designing those systems from first principles at Amazon and DoorDash maps directly." Frame your work as a template for what Netflix Ads needs to build.

**Preparation:**
- Ad inventory forecasting: predicting available impression supply by targeting segment — this is a time-series + uncertainty quantification problem
- Ad pacing under inventory constraints: same Lagrangian relaxation framing as budget pacing at DoorDash
- Product-mindedness still required: tie every design to subscriber retention (ads must not degrade watch experience) — this tension is unique to Netflix vs. pure-play ad platforms
- Know their general ML stack: Metaflow-style DAG orchestration, real-time inference SLA requirements
- Netflix gotcha: Monitoring is not an afterthought. Proactively say "Here's how I'd know it's working in prod: shadow scoring for 48h, holdout group sized at X%, alert on P(drift) > threshold."

#### Roblox
**Ads/platform economics context (verified):** Roblox launched rewarded video ads (6–30s, opt-in for 13+ users) in 2025, integrated with Google Ad Manager, and is under investor pressure to prove ad revenue can scale without degrading engagement. Their Q3 2025 earnings explicitly discussed advertising/monetization strategy as a key growth lever. Users 13+ grew from 40M (Q3 2023) to 101M (Q3 2025) — the addressable ad audience is now majority adult.

**Your angle:** "Roblox's ads challenge is a marketplace design problem — how do you price ad inventory in a platform where the supply (developer attention) and demand (advertiser budgets) are both price-sensitive? My mechanism design background applies directly to the opt-in rewarded format: you're designing an exchange where users, developers, and advertisers all have strategic incentives."

**Preparation:**
- Rewarded ad format ML: user-side value model (will this user opt in?), advertiser-side ROI model, developer-side revenue optimization — a three-sided marketplace
- Safety/content moderation ML: Roblox processes 6.1B chat messages/day with ML classifiers (PII detection, 98% recall) — expect content safety system design questions
- Platform economics framing: ads must not cannibalize the creator economy (developer earnings) — discuss the mechanism design tension explicitly
- User age distribution still matters: targeting, consent, and content restrictions for under-13 users require a separate serving path

### Cluster D: Foundation Model / Research-Adjacent — OpenAI
**Common thread:** Deep learning fundamentals, training infrastructure, alignment, safety
**Your delta:** Largest — requires breadth in modern DL that your ads background may not cover
**Preparation (focused, not exhaustive):**
- Know transformer architecture cold: attention is all you need, positional encoding, KV cache, flash attention
- RLHF pipeline: SFT → reward model training → PPO/DPO fine-tuning. Know why DPO is simpler than PPO.
- Scaling laws: Chinchilla (compute-optimal training), inference-time compute (test-time scaling)
- Evals: how you measure model capability, benchmark saturation problem, red-teaming
- Your advantage: multi-agent systems (multi-agent RL, emergent behavior, coordination games) is a genuine research direction at OpenAI
- Behavioral bar: OpenAI values intellectual honesty and deep thinking over polish. Bring specific opinions on hard problems (e.g., "I think reward hacking in RLHF is fundamentally a distributional shift problem, not just a specification problem, because...")

---

## 4. Weekly Execution Plan (10 Weeks)

### Phase 1: Foundation (Weeks 1–3)
**Goal:** Lock in coding fundamentals + polish core ML system designs

| Day type | Activity | Time |
|---|---|---|
| Coding (4x/week) | 2 LeetCode medium/hard per session; focus on Graphs, DP, Heaps | 2h/session |
| System Design (2x/week) | Design one system end-to-end; write it up; self-evaluate | 2h/session |
| Behavioral (1x/week) | Draft + refine 3 anchor STAR stories | 1.5h/session |

**Week 1:** Graphs (BFS/DFS/topological sort) + Design: Two-Tower Retrieval System
**Week 2:** DP (1D, 2D, interval) + Design: Ads Auction System End-to-End
**Week 3:** Heaps, Priority Queues, Sliding Window + Design: Real-Time Feature Store

**Deliverable:** 30 LeetCode solved; 3 system design docs written; behavioral stories drafted

---

### Phase 2: Company Clusters (Weeks 4–7)
**Goal:** Domain-specific depth for each cluster

| Week | Focus Cluster | Coding Theme | System Design | Behavioral Focus |
|---|---|---|---|---|
| 4 | Ads/Marketplace (TTD, Meta, Uber) | Rate limiters, task schedulers, online algorithms | Autobidding system + budget pacing | Impact story from DoorDash Ads |
| 5 | Practical Engineering (Stripe, Shopify) | API design, debugging exercises, LRU/rate limiter implementations | ML inference pipeline with SLAs | Technical judgment / hard trade-off story |
| 6 | Personalization + Ads (Netflix, Roblox) | Graph problems (social network, recommendation graph) | Recommendation system + Ad inventory forecasting + A/B test platform | Cross-functional influence story |
| 7 | Foundation Models (OpenAI) | Python implementations (attention, tokenizer, BPE) | LLM serving infrastructure (KV cache, batching) | Intellectual honesty / research opinion |

---

### Phase 3: Mock Interviews + Gap Close (Weeks 8–9)
**Goal:** Simulate real interview conditions; identify and fix weaknesses

- 2 mock coding interviews per week (use Pramp, interviewing.io, or a trusted peer)
- 1 mock ML system design per week — record yourself, review for filler, vagueness, missing monitoring
- 1 mock behavioral per week — focus on conciseness (STAR in <3 minutes)
- Gap close: If coding speed is low → drill patterns. If system design is shallow → re-read relevant engineering blogs.

---

### Phase 4: Company-Specific Sprints (Week 10)
**Goal:** Tailor final prep to confirmed interview pipeline

- Research the company's recent engineering blog posts (last 12 months)
- Review their open-source contributions (e.g., Meta's FAISS, Uber's Cadence/Michelangelo papers)
- Prepare 2–3 specific questions that show you've read their systems ("I saw you moved from GSP to first-price auction — how did you handle advertiser learning period?")
- Final behavioral polish: tailor stories to the company's culture values

---

## 5. Key Resources

### Papers (Must-Read, Prioritized)

| Paper | Company Relevance | Why |
|---|---|---|
| "Ad Click Prediction: a View from the Trenches" (Google, 2013) | Meta, TTD | FTRL, feature engineering at scale |
| "Deep Learning Recommendation Model (DLRM)" (Meta, 2019) | Meta | Architecture they use in production |
| "Scaling Distributed Machine Learning with the Parameter Server" (CMU/Baidu) | Meta, Uber | Distributed training fundamentals |
| "Attention Is All You Need" (Google, 2017) | OpenAI | Transformer baseline |
| "Training language models to follow instructions with human feedback" (OpenAI, 2022) | OpenAI | RLHF pipeline |
| "Direct Preference Optimization" (Stanford, 2023) | OpenAI | DPO as simpler RLHF |
| "Real-time Personalization using Embeddings for Search Ranking at Airbnb" (KDD 2018) | Netflix, Uber, Roblox | Embedding-based retrieval |
| "Michelangelo: Uber's Machine Learning Platform" (Uber, 2017) | Uber | Feature store, training, serving |
| "Chinchilla" / "Scaling Laws for Neural Language Models" (DeepMind/OpenAI) | OpenAI | Compute-optimal training intuition |
| "Budget Pacing for Targeted Online Advertisements at LinkedIn" (KDD 2014) | TTD, Meta | Your domain — know this deeply |
| "Autobidding with Constraints" (Google, 2021) | TTD, Meta, DoorDash | Direct relevance to your work |

### Engineering Blogs (Bookmark and Read Last 12 Months)

| Blog | Key Posts to Find |
|---|---|
| TTD Engineering (thetradedesk.com/engineering) | Unified ID, Koa AI, RTB infrastructure |
| Meta AI / Meta Engineering | Ads ranking, DLRM, recommendation systems |
| Uber Engineering (eng.uber.com) | ETA improvements, Michelangelo, surge pricing |
| Netflix Tech Blog (netflixtechblog.com) | Recommendation, Metaflow, A/B testing, Ads |
| Stripe Engineering (stripe.com/blog/engineering) | Fraud detection, ML at Stripe, reliability |
| Shopify Engineering (shopify.engineering) | ML for e-commerce, infrastructure |
| OpenAI Research (openai.com/research) | GPT-4 system card, alignment updates |
| Google Research (on auctions) | First-price auction transition, bid shading |

### Frameworks & Tools to Know by Name (not deep expertise required)

| Tool | Context |
|---|---|
| FAISS / ScaNN | Approximate nearest neighbor for embedding retrieval (Meta, Netflix) |
| Flink / Spark | Streaming vs. batch feature pipelines |
| Ray / Metaflow | ML workflow orchestration (Netflix, Uber) |
| Triton Inference Server | GPU-optimized model serving (OpenAI adjacent) |
| Feature stores: Feast, Tecton, Hopsworks | Know the online/offline consistency problem |
| Pydantic v2 | Request/response schema definition (Stripe, Shopify) |
| H3 (Uber) | Hexagonal geospatial indexing for location features |

### LeetCode Problem Sets (Curated for This Target List)

| Category | Problems |
|---|---|
| Graphs | Course Schedule II, Clone Graph, Word Ladder, Network Delay Time, Cheapest Flights K Stops |
| DP | Coin Change, Word Break, Edit Distance, Burst Balloons, Best Time to Buy/Sell Stock variants |
| Heaps | Merge K Sorted Lists, Top K Frequent Elements, Task Scheduler, Find Median from Data Stream |
| System Design Coding | LRU Cache, Rate Limiter, Design Twitter Feed, Design Search Autocomplete |
| Ads/Auction Specific | Median of Data Stream (bid price tracking), Meeting Rooms II (budget slot allocation) |

---

## Execution Checklist

- [ ] Polish 30-second intro pitch incorporating mechanism design angle
- [ ] Write 3 anchor STAR stories (scope: org-level impact)
- [ ] Solve 60+ LeetCode problems across priority categories
- [ ] Write 6 system design docs (auction, two-tower, feature store, recommendation, A/B platform, LLM serving)
- [ ] Read 5+ engineering blog posts per target company
- [ ] Complete 4+ mock interviews (2 coding, 1 system design, 1 behavioral)
- [ ] Prepare 2–3 company-specific questions per company showing blog/paper familiarity
