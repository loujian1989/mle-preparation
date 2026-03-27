# MLE Interview Prep: 10-Week Strategy for Staff/Senior Roles
**Target Companies:** Stripe, Uber, Meta, OpenAI, Shopify, Netflix, Roblox
**Background:** PhD CS (game theory/mechanism design), DoorDash Ads (auction/pricing/autobidding), Amazon (Sponsored Products)

---

## 0. Verified Job Listings + Interview Intel (Research as of March 2026)

> All JDs verified live. Netflix Ads-specific JD links returned 404 at time of research (roles may have been filled/rotated — check explore.jobs.netflix.net for current postings). All other listings confirmed active.

---

### Stripe

**Titles:** MLE (multiple tracks), SWE ML Infrastructure

| Role | Level | URL |
|---|---|---|
| MLE, Payments ML Accelerator | Senior (7+ yrs) | stripe.com/jobs/listing/…/7079044 |
| MLE, Foundation Model | Senior | stripe.com/jobs/listing/…/7275014 |
| MLE, Stripe Assistant | Mid-Senior | stripe.com/jobs/listing/…/7629052 |
| Senior MLE, Stripe Assistant | Senior | stripe.com/jobs/listing/…/6894964 |
| SWE, ML Infrastructure | Senior | stripe.com/jobs/listing/…/7528260 |
| PhD MLE, New Grad | New Grad | stripe.com/jobs/listing/…/7216668 |

**Best fit for you:** MLE Payments ML Accelerator — fraud detection, authorization optimization, deep learning + LLMs, 7+ years required, $212K–$318K.

**JD Key Requirements (Payments ML Accelerator, verified):**
- 7+ years end-to-end ML development + production deployment
- Python, Scala, Spark
- Deep learning + LLM/foundation model expertise
- Focus: fraud detection → authorization optimization, across merchants/issuers/customers

**Interview Process (from Glassdoor + LeetCode discuss + Taro, 2024–2025):**
1. Recruiter call (30 min)
2. OA — CoderPad, ~45 min, semi-real-life data manipulation with conditional logic (not LeetCode)
3. ML coding round — given a dataset in Jupyter, build + evaluate a model in 1 hour; construct target variable, feature selection, beat a random baseline
4. Onsite (4 rounds): coding, debugging (clone a repo + fix bugs), ML system design, hiring manager
- Total ~17 days average to hire

**Key insider notes:**
- The debug round is unique to Stripe — practice reading production Python code and diagnosing failures systematically
- "No LeetCode" — they provide real data and expect working, evaluated models
- 27% positive experience on Glassdoor — process is demanding and somewhat opaque; recruiter communication can be slow
- ML system design at Stripe: expect fraud detection architecture, recommendation for merchants, authorization optimization

---

### Uber

**Titles:** MLE (various teams), Staff MLE

| Role | Level | URL |
|---|---|---|
| Senior MLE - Marketplace Pricing | Senior | uber.com/global/en/careers/list/145740 |
| Staff MLE - Marketplace Pricing & Incentives | Staff | uber.com/global/en/careers/list/140494 |
| MLE II - Pricing | Mid | uber.com/global/en/careers/list/146338 |
| MLE II - Optimization | Mid | uber.com/global/en/careers/list/145150 |
| Staff MLE - Applied AI | Staff | uber.com/global/en/careers/list/146989 |

**Best fit for you:** Senior MLE Marketplace Pricing + Staff MLE Marketplace Pricing & Incentives

**JD Key Requirements (Senior MLE Marketplace Pricing, verified):**
- 4+ years deploying ML models in production
- Python/Scala/Java/Go + Spark/Ray/Flink
- DNNs, multi-task models, transformers, mathematical optimization
- "Marketplace pricing algorithms, RL, causal ML" preferred
- Real-time multi-objective optimizations at 1M+ predictions/second
- Comp: $202K–$224K base (NY)

**Interview Process (from InterviewQuery, Glassdoor, 2024–2025):**
1. Resume screen + recruiter call
2. Technical screen: 1–2 rounds, LeetCode-style coding + ML fundamentals
3. Virtual onsite (4–5 rounds, 45–60 min each):
   - Round 1: Coding & Data — Python, algorithmic + data manipulation, edge cases
   - Round 2: Applied ML — end-to-end modeling, feature selection, metric choice, monitoring
   - Round 3: ML System Design — real-time inference architecture, failure modes, observability
   - Round 4: Product & Collaboration — cross-functional communication, business impact framing
   - Round 5: Behavioral — accountability, setbacks, ownership
- Total timeline: 3–6 weeks

**Key insider notes:**
- Marketplace thinking is explicitly scored: "how do prediction errors cascade through pricing, supply, driver incentives"
- Bring production incident stories — they ask about operational impacts, not just model metrics
- Graph algorithms appear in coding (word transformation, graph traversal)
- Rolling metrics computation is a common coding question

---

### Meta

**Title:** Software Engineer, Machine Learning (not "MLE Engineer")

| Role | Level | URL |
|---|---|---|
| Software Engineer, Machine Learning | E5/E6 | metacareers.com/profile/job_details/1436181490732782 |

**JD Key Requirements (verified):**
- 6+ years programming OR 3+ years + PhD
- PyTorch/TF/Python/C++/Java
- 2+ years in ML, recommendation systems, pattern recognition, data mining, or AI
- "Developing ML models at scale from inception to business impact"
- Ads prediction is a primary focus area alongside Feed/Reels ranking

**Interview Process (from Glassdoor + IGotAnOffer + Medium, 2024–2025):**
1. Phone screen: 1 medium-level LeetCode
2. Onsite (4–6 rounds):
   - 2 coding rounds (LeetCode medium/hard — this is the hardest coding bar of the 8 companies)
   - 1 ML system design: designing feed ranking, ads prediction, or recommendation systems end-to-end
   - 1 behavioral: scope of impact, ambiguity, cross-functional influence
- Evaluation axes: correctness, runtime/space complexity, communication of trade-offs

**Key insider notes:**
- "Coding bar is closer to a pure SWE loop than most MLE roles" — this is the most LeetCode-heavy of the 8
- System design focus: Feed ranking, ads CTR prediction, recommendation systems
- Ads ML design questions: "design an evaluation framework for ads ranking" is a documented question
- High bar for E6 (Staff): expect scope-of-impact stories at org/company level, not team level

---

### OpenAI

**Titles:** MLE (Integrity), Research Engineer (Applied AI, Post-Training, AI for Science), SWE Model Inference

| Role | Level | URL |
|---|---|---|
| MLE, Integrity | Mid-Senior | openai.com/careers/machine-learning-engineer-integrity-san-francisco |
| Research Engineer, Applied AI Engineering | Senior | openai.com/careers/research-engineer-applied-ai-engineering-san-francisco |
| Research Engineer / Research Scientist, Post-Training | Senior | openai.com/careers/research-engineer-research-scientist-post-training-san-francisco |
| SWE, Model Inference | Senior | openai.com/careers/software-engineer-model-inference-san-francisco |

**Best fit for you:** MLE Integrity (adversarial ML, game theory angle) + Research Engineer Applied AI

**Interview Process (from OpenAI official guide + InterviewQuery + Medium, 2024–2025):**
1. Recruiter screen (30–45 min) — career trajectory, OpenAI mission familiarity, team fit
2. ML coding interview (45–60 min) — CoderPad, real-world engineering: model architectures, training methodologies, gradient descent, production considerations; NOT abstract algorithms
3. Virtual onsite (4–6 hours over 1–2 days):
   - Complex coding
   - ML system design (safety systems, recommenders, topic classification at scale)
   - Project deep-dive / presentation (defend a "Learning Sample" take-home)
   - AI ethics discussion
   - Behavioral
4. Hiring committee (~1 week)
- Total timeline: 6–8 weeks

**Sample questions (verified from community):**
- Interpolating missing data with Pandas groupby + interpolate
- Logistic regression on separable data (fails to converge — requires recognizing regularization fix)
- Design an unsafe content detection pipeline with bias monitoring
- Scalable recommender system with distributed computing

**Key insider notes:**
- They value intellectual honesty + mission alignment above polish — have an opinion on hard alignment problems
- Production debugging is stressed: exploding gradients, vanishing signals, batch norm instability
- "Learning Sample" (take-home defended in interview) is a differentiating element — prepare to walk through a real project end-to-end
- AI ethics discussion is real — not perfunctory; have a formed position on RLHF trade-offs, reward hacking, and dual-use risks

---

### Shopify

**Titles:** Applied MLE, MLE (various tracks)

| Role | Level | URL |
|---|---|---|
| Applied Machine Learning Engineers | Mid-Senior | shopify.com/careers/applied-machine-learning-engineers_19b9dea6 |
| **MLE - Ads** | Senior | shopify.com/careers/machine-learning-engineer-ads_5485d8c3 |
| MLE - HSTU | Senior | shopify.com/careers/machine-learning-engineer-hstu_04b84b82 |
| Applied ML Engineering - GenAI, AI Agent | Senior | shopify.com/careers/applied-ml-engineering-genai-ai-agent_dae1b282 |
| MLE - Search | Senior | shopify.com/careers/machine-learning-engineer-search_c15b011d |
| MLE Infrastructure | Senior | shopify.com/careers/machine-learning-infrastructure-engineers_896a7d5f |

**Best fit for you:** MLE - Ads (direct match to your domain)

**Interview Process (from InterviewQuery + Glassdoor + Taro, 2024–2025):**
1. Recruiter screen
2. Life Story interview (30–60 min) — cultural values alignment, resilience, growth mindset; assessed BEFORE technical rounds
3. ML Technical Screen — CoderPad, real-world commerce data (merchant transactions, product metadata); collaborative filtering, GBDTs, inference optimization; passing threshold 7.5+
4. Take-home (4–6 hrs) — containerized FastAPI service or documented notebook; SHAP interpretability, time-series validation splits, anonymized merchant data
5. Onsite loop — real-time ML system design (<100ms inference), feature pipeline via Pano (Shopify's Feast-based feature store), Kafka, Black Friday resilience, multi-region deployment, A/B test design for GMV/conversion
- Acceptance rate: ~0.3% (extremely competitive)

**Key insider notes:**
- The Life Story round comes BEFORE technical — it's a real gate, not a formality; align your narrative to merchant-first impact
- They use Pano (Feast-based feature store) + Ray + Kafka — name these explicitly in system design
- "Move fast, communicate clearly, own decisions" — every design decision needs a stated trade-off
- MLE - Ads is a direct domain match: bring your DoorDash Ads experience and connect it to merchant GMV impact (advertisers on Shopify = merchants)
- Black Friday resilience is a real interview topic — design for 10x traffic spikes, failover, degraded-mode serving

---

### Netflix

**Titles:** Machine Learning Engineer (L5, L5 Senior)

| Role | Level | URL |
|---|---|---|
| MLE L5 - Ads | L5 (Senior) | explore.jobs.netflix.net — search "MLE Ads" |
| MLE L5 Senior - Ads Inventory Management & Forecasting | L5 Senior | explore.jobs.netflix.net — search "Inventory Forecasting" |
| MLE L5 - Content and Studio | L5 | jobs.netflix.com/jobs/308370003 |

> Note: Specific Ads JD direct links returned 404 at research time — roles confirmed active via moaijobs.com and earlier search results. Check explore.jobs.netflix.net directly.

**JD Key Requirements (from earlier verified search, Dec 2025):**
- ML for ad ranking, pacing, and personalization (L5 Ads)
- Predictive models for advertising campaign effectiveness + campaign delivery forecasting (L5 Senior - Inventory)
- Low-latency real-time ad systems — productionized predictive model deployment
- Metaflow-style orchestration background valued

**Interview Process (from Blind + InterviewQuery + Exponent, 2024–2025):**
1. Recruiter screen — background, motivation, Freedom & Responsibility culture fit
2. Technical screen — take-home modeling quiz (feature engineering, model evaluation) + live Python coding (algorithmic)
3. Virtual onsite (3 rounds):
   - ML system design: online training pipelines, ad-break prediction, recommendation architecture; latency budgets (<200ms) required
   - Algorithmic coding: array manipulation, DP, bit operations, linked lists; medium to hard
   - Behavioral: collaboration, previous project deep-dive
4. For L5: confirmed as "1 coding, 1 previous experience, 2 director-level conversations" including cross-team partner director
- Comp: $455K–$525K median total; base $200K–$631K range

**Key insider notes:**
- Director-level conversations at L5 are real — they assess whether you can communicate technical decisions at exec level, not just team level
- "Ad-break prediction" has been cited as an actual system design question — prepare this specifically
- Netflix ghosting after friendly interviews is a documented complaint — don't assume silence = good news
- Freedom & Responsibility culture: expect to be asked how you operate autonomously without process guardrails

---

### Roblox

**Titles:** Senior MLE, Principal MLE

| Role | Level | URL |
|---|---|---|
| [2026] Senior MLE, AI Platform | Senior (PhD early career) | careers.roblox.com/jobs/7403998 |
| [2026] Senior MLE, Recommendation Systems | Senior (PhD early career) | careers.roblox.com/jobs/7350081 |
| [2026] Senior MLE, NLP | Senior (PhD early career) | careers.roblox.com/jobs/7324377 |
| [2026] Senior MLE, Multimodal AI / CV | Senior (PhD early career) | careers.roblox.com/jobs/7323437 |
| Principal MLE, User Connections | Principal | careers.roblox.com/jobs/6268817 |

**Best fit for you:** Senior MLE Recommendation Systems + Principal MLE User Connections

**JD Key Requirements (from search, verified):**
- PhD in CS or related field with expertise in recommender systems, search, information retrieval, or generative models
- End-to-end ML systems including data analysis and model deployment
- ML algorithms, data pipelines

**Interview Process (from InterviewQuery + Glassdoor + Taro, 2024–2025):**
1. Online assessment (HackerRank/CodeSignal) — 2 standard technical rounds
2. ML system design round — designing a recommendation system is a documented question
3. Project deep-dive + behavioral
4. Additional round: team/culture fit
- Total: 4–6 weeks; 5 rounds for Senior MLE
- Process is decentralized — hiring managers customize by team

**Key insider notes:**
- Interviewers described as friendly; process is more structured than OpenAI/Netflix
- Recommendation system design is the anchor ML system design question — prepare this cold
- Platform economics / ads framing is an emerging angle (see Cluster C notes) but not yet the primary interview focus — recommendation + safety is still the dominant lens
- Principal MLE User Connections is a strong target given your multi-agent/network modeling background

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

### Cluster A: Ads/Marketplace ML — Meta, Uber
**Common thread:** Real-time bidding, two-sided marketplace, high-throughput feature serving
**Your delta:** Minimal — this is your core domain
**What to add:**
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
| 4 | Ads/Marketplace (Meta, Uber) | Rate limiters, task schedulers, online algorithms | Autobidding system + budget pacing | Impact story from DoorDash Ads |
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
| "Ad Click Prediction: a View from the Trenches" (Google, 2013) | Meta | FTRL, feature engineering at scale |
| "Deep Learning Recommendation Model (DLRM)" (Meta, 2019) | Meta | Architecture they use in production |
| "Scaling Distributed Machine Learning with the Parameter Server" (CMU/Baidu) | Meta, Uber | Distributed training fundamentals |
| "Attention Is All You Need" (Google, 2017) | OpenAI | Transformer baseline |
| "Training language models to follow instructions with human feedback" (OpenAI, 2022) | OpenAI | RLHF pipeline |
| "Direct Preference Optimization" (Stanford, 2023) | OpenAI | DPO as simpler RLHF |
| "Real-time Personalization using Embeddings for Search Ranking at Airbnb" (KDD 2018) | Netflix, Uber, Roblox | Embedding-based retrieval |
| "Michelangelo: Uber's Machine Learning Platform" (Uber, 2017) | Uber | Feature store, training, serving |
| "Chinchilla" / "Scaling Laws for Neural Language Models" (DeepMind/OpenAI) | OpenAI | Compute-optimal training intuition |
| "Budget Pacing for Targeted Online Advertisements at LinkedIn" (KDD 2014) | Meta | Your domain — know this deeply |
| "Autobidding with Constraints" (Google, 2021) | Meta, DoorDash | Direct relevance to your work |

### Engineering Blogs (Bookmark and Read Last 12 Months)

| Blog | Key Posts to Find |
|---|---|
| Meta AI / Meta Engineering | Ads ranking, DLRM, recommendation systems |
| Uber Engineering (eng.uber.com) | ETA improvements, Michelangelo, surge pricing |
| Netflix Tech Blog (netflixtechblog.com) | Recommendation, Metaflow, A/B testing |
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

---

## Appendix: Broader Company Search (Research as of March 2026)

> Companies beyond the 7 primary targets. Filtered for: Seattle office or fully remote US; ads/marketplace ML domain fit; Staff/Senior+ level. Ranked by fit quality.

---

### Tier 1 — Apply Immediately (Direct Domain Match)

#### Reddit

**Best role:** Staff MLE, Ads Auction (Ads Marketplace Quality) — Fully Remote US

| Role | Level | URL |
|---|---|---|
| Staff MLE, Ads Auction (Ads Marketplace Quality) | Staff | job-boards.greenhouse.io/reddit/jobs/7181821 |
| Staff MLE, Bidding and Pacing (Advertiser Optimization) | Staff | job-boards.greenhouse.io/reddit/jobs/7074763 |
| Senior Staff MLE, Bidding and Pacing | Senior Staff | job-boards.greenhouse.io/reddit/jobs/6158103 |

**Compensation:** $230K–$322K base + equity (RSUs)

**JD Key Requirements (Ads Auction, verified):**
- 8+ years industry experience; 5+ years focused on data-driven marketplace optimization at scale
- Ads marketplace optimization, auction/pricing mechanisms
- Spark, Kafka, Beam, Flink; TensorFlow/PyTorch; Java, Python, Golang, C++
- "Lead strategy for ads marketplace, auction and pricing initiatives"
- "Develop algorithms improving marketplace efficiency and auction optimization"

**Why it's a top pick:**
- Fully remote US — strongest location match for Seattle
- Ads Auction + Bidding & Pacing roles are *exact* descriptions of your DoorDash work (TCPA, TROAS, Maximize Conversions, cost caps, campaign budget optimization, real-time bidding)
- Reddit is a growth-stage ads platform (ads revenue scaling from near-zero to core business) — same inflection point as DoorDash Ads 2-3 years ago
- Staff-level scope: quarterly planning ownership, cross-functional technical leadership

**Interview Process (from Glassdoor/Blind, 2024–2025):**
1. Recruiter screen
2. Technical phone screen — ML fundamentals + coding
3. Onsite (4–5 rounds): coding (LeetCode medium), ML system design, ads domain knowledge, behavioral

**Your mechanism design angle:**
> "Reddit's ads platform is at the same inflection point DoorDash Ads was — moving from simple CPC to full autobidding with CPA/ROAS objectives. The auction design choices made now (first-price vs. second-price, pacing mechanism, advertiser learning period) determine long-term marketplace health. I've been the person who made those calls at DoorDash."

---

#### Pinterest

**Best role:** Sr. MLE / Economist, Ads Marketplace — Seattle-commutable (SF, Palo Alto, or Seattle; 1-2x/month in-office)

| Role | Level | URL |
|---|---|---|
| Sr. MLE / Economist, Ads Marketplace | Senior/Staff | pinterestcareers.com/jobs/7195341 |
| Staff MLE, Ads Marketplace | Staff | pinterestcareers.com/en/jobs/4981431 |
| Principal MLE, Ads Delivery | Principal | pinterestcareers.com/jobs/6963868 |
| Staff MLE, Monetization | Staff | pinterestcareers.com/jobs/6133257 |

**Compensation:** ~$180K–$280K base (Senior/Staff range, inferred from levels.fyi)

**JD Key Requirements (Sr. MLE / Economist, Ads Marketplace, verified):**
- MS or PhD in CS, Economics, Statistics, or related field (required at Senior level)
- "Knowledge in auction theory, market design, and econometrics"
- "Industry experience applying economics or ML to ads auctions, pricing, marketplaces"
- A/B testing, causal inference, online experimentation at large scale
- "Implement auctions, tune marketplace parameters (utility function), model long-term effects"
- Focus areas: ranking/pricing/mechanism design, bidding/budgeting innovation, advertiser churn/retention modeling

**Why it's a top pick:**
- The role title literally says "Economist" — your PhD in computational game theory + mechanism design is the explicit qualification, not a bonus
- Seattle commutable (1-2x/month) — viable for Seattle-based candidate
- Pinterest's ad auctions interact with organic content ranking (promoted pins, shopping ads) in ways that require formal mechanism design thinking — second-order effects, ad fatigue, long-term marketplace health
- JD explicitly calls out "model long-term effects to reduce ad fatigue" and "anticipate second-order effects for new ad offerings" — PhD-level systems thinking required

**Interview Process (from Glassdoor/Blind/Exponent, 2025):**
1. Recruiter screen
2. Coding screen (LeetCode medium — Python)
3. Onsite (4–5 rounds, 2-4 weeks total):
   - 2 ML rounds (1 domain-specific for Ads, 1 general systems)
   - 2 coding rounds (LeetCode medium)
   - 1 behavioral
- Stack: Python, PyTorch, SQL, Spark, Hadoop, Docker/Kubernetes

**Your mechanism design angle:**
> "Pinterest's ads marketplace has a structural challenge that pure e-commerce platforms don't: you're running an auction over visual content where organic and paid signals interact. The mechanism design question isn't just 'maximize revenue' — it's 'design an auction where promoted content doesn't degrade the organic discovery experience.' That's a constrained multi-objective optimization problem I've modeled formally."

---

### Tier 2 — Consider if Open to Relocation

#### Snap

**Best role:** Principal MLE, Ads Marketplace — Palo Alto HQ (not Seattle/remote — listed for awareness)

| Role | Level | URL |
|---|---|---|
| Principal MLE, Ads Marketplace | Principal | wd1.myworkdaysite.com/recruiting/snapchat/snap — search "Principal MLE Ads Marketplace" |

**Compensation:** $480K median total (levels.fyi)

**Why it's listed:** Strong domain match (high-throughput RTB auction, DL-based ad ranking at scale, Principal-level scope). Only consider if open to Palo Alto relocation or remote negotiation.

**Interview note:** No dedicated behavioral round — company values ("Kind, Smart, Creative") evaluated across all rounds; team matching after offer takes weeks.

---

### Skip List

| Company | Reason to Skip |
|---|---|
| Expedia | Comp too low ($184K–$258K base) relative to other targets |
| Lyft | Comp too low ($176K–$220K), hybrid Seattle 3 days/week, smaller marketplace scope |
| Amazon DSP | Prior employer — evaluate separately based on team/manager knowledge |
| Google/YouTube Seattle | 3–6 month process, LeetCode-heaviest bar, low process ROI unless specifically targeting Google |
| TikTok/ByteDance | Geopolitical risk (US ban proceedings ongoing), limited Seattle presence |

---

### Summary Recommendation

| Priority | Company | Role | Location | Est. Base | Domain Fit |
|---|---|---|---|---|---|
| P0 | Reddit | Staff MLE Ads Auction | Fully Remote | $230K–$322K | ★★★★★ |
| P0 | Pinterest | Sr. MLE/Economist Ads Marketplace | Seattle-commutable | ~$200K–$260K | ★★★★★ |
| P2 | Snap | Principal MLE Ads Marketplace | Palo Alto only | ~$300K+ total | ★★★★☆ |

**Recommended action:** Add Reddit and Pinterest to the active interview pipeline alongside the 7 primary targets. Both are mechanism design / ads auction exact fits with favorable location policies.
