# MLE Interview Prep: 10-Week Strategy for Staff/Senior Roles
**Target Companies:** Stripe, Uber, Meta, OpenAI, Shopify, Netflix, Roblox, Reddit, Pinterest, Whatnot, Microsoft
**Background:** PhD CS (game theory/mechanism design), DoorDash Ads (auction/pricing/autobidding), Amazon (Sponsored Products)

---

## 0. Verified Job Listings + Interview Intel (Research as of March 2026)

> All JDs verified live. Netflix target role updated July 2026: Research Scientist 5, Marketplace - Ads DSE (previous listing filled — new listing active). Microsoft Principal Applied Scientist (Auction & Bidding) added July 2026 — active listing. All other listings confirmed active or flagged where status is uncertain.

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

**Target role:** Research Scientist 5 (RS5), Marketplace - Ads DSE *(updated July 2026)*

| Role | Level | URL |
|---|---|---|
| Research Scientist 5, Marketplace - Ads DSE | RS5 ≈ Staff | [Netflix Careers](https://explore.jobs.netflix.net/careers?pid=790315989769&domain=netflix.com&sort_by=relevance) |

**JD Key Requirements (verified July 2026):**
- PhD or MS in CS, Statistics, Mathematics, or quantitative field
- Python, Scala, or Java
- Deep expertise in ML, optimization, and data analysis applied to ad tech (targeting, ranking, bidding) — explicitly required
- Experience prototyping algorithms on production-scale data
- Must translate technical results into business outcomes (strong business acumen required)

**Interview Process (3 rounds, 3–5 weeks):**
1. High-level technical discussion with hiring manager — background, research fit, culture
2. Deep-dive technical screenings — ML fundamentals, optimization algorithms, production deployment; expect first-principles math derivations (not API-level answers)
3. Virtual onsite panel — ML system design + **research deep-dive**: present your own past work/projects, defend methodology, discuss how it translates to Netflix Ads problems
- **Key difference from MLE loop:** heavier research framing throughout; production accountability still required (you must demonstrate deployed impact, not just research output)
- Comp (RS5): $466K–$750K (salary + stock options annually; no bonus structure)

**Key insider notes:**
- "Ad-break prediction" has been cited as an actual system design question — prepare this specifically
- Netflix ghosting after friendly interviews is a documented complaint — don't assume silence = good news
- Freedom & Responsibility culture: expect to be asked how you operate autonomously without process guardrails
- Monitoring is not an afterthought — proactively say "here's how I'd know it's working in prod: shadow scoring for 48h, holdout group sized at X%, alert on P(drift) > threshold"

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

### Reddit

**Titles:** Staff MLE (Ads Engineering)

| Role | Level | URL |
|---|---|---|
| Staff MLE, Ads Auction (Ads Marketplace Quality) | Staff | job-boards.greenhouse.io/reddit/jobs/7181821 |
| Staff MLE, Bidding and Pacing (Advertiser Optimization) | Staff | job-boards.greenhouse.io/reddit/jobs/7074763 |
| Senior Staff MLE, Bidding and Pacing | Senior Staff | job-boards.greenhouse.io/reddit/jobs/6158103 |

**Best fit for you:** Staff MLE Ads Auction — directly owns auction/pricing strategy. Staff MLE Bidding and Pacing owns TCPA/TROAS/autobidding (exact DoorDash analogue). Fully remote US. Comp: $230K–$322K base + RSUs; MLE IC median total $506K (levels.fyi).

**JD Key Requirements (Staff MLE Ads Auction, verified):**
- 8+ years industry experience; 5+ years focused on data-driven marketplace optimization at scale
- Ads marketplace optimization, auction/pricing mechanisms
- Spark, Kafka, Beam, Flink; TensorFlow/PyTorch; Java, Python, Golang, C++
- "Lead strategy development, quarterly planning, and execution for ads marketplace, auction and pricing"
- "Develop algorithms improving marketplace efficiency and auction optimization"

**Interview Process (from Glassdoor/Blind/InterviewQuery/Exponent, 2024–2025):**
1. Recruiter call
2. Phone screen (45 min) — **live ML model building from a real dataset** (NOT LeetCode); Google allowed but algorithmic thinking is evaluated; tasks: data cleaning, feature engineering, model selection, evaluation, optimization
3. Onsite (6 hours with breaks, virtual):
   - ML system design — domain-specific to Reddit (documented: stock prediction from Reddit comments, recommendation systems; ads-specific not publicly documented)
   - Generic system design round
   - Pair programming / coding round
   - Behavioral + cross-functional round
   - Hiring manager round
- Difficulty: 2.7/5 (Glassdoor) — relatively accessible bar vs. Meta/OpenAI
- Total timeline: 2–6 weeks

**Key insider notes:**
- Phone screen is practical ML (not LeetCode) — practice building a working model on a dataset in 45 min: target variable construction, feature selection, evaluation, narrate trade-offs
- ML system design is described as "fun" and domain-specific — expect problems grounded in Reddit's actual product (content ranking, spam, ads)
- Growth-stage ads platform: Reddit is moving from simple CPC → full autobidding with CPA/ROAS objectives — same inflection DoorDash Ads went through ~2 years ago
- Headcount risk: one documented case of offer rescinded due to headcount re-prioritization after a positive process — confirm headcount status with recruiter
- 33% positive experience on Glassdoor (below average) — process quality is inconsistent; follow up proactively
- Stack: Kafka (65B+ daily events), Flink (real-time stream processing), Spark (batch), TF/PyTorch; control theory + RL for autobidding/pacing; products: Lowest Cost, Cost Caps, Campaign Budget Optimization, TCPA, TROAS

---

### Pinterest

**Titles:** MLE / Economist (Ads Marketplace), Staff MLE (Monetization), Principal MLE (Ads Delivery)

| Role | Level | URL |
|---|---|---|
| Sr. MLE / Economist, Ads Marketplace | IC15 (Senior) | pinterestcareers.com/jobs/7195341 |
| Staff MLE, Ads Marketplace | IC16 (Staff) | pinterestcareers.com/en/jobs/4981431 |
| Principal MLE, Ads Delivery | Principal | pinterestcareers.com/jobs/6963868 |
| Staff MLE, Monetization | IC16 (Staff) | pinterestcareers.com/jobs/6133257 |

**Best fit for you:** Sr. MLE / Economist, Ads Marketplace — role title explicitly requires auction theory + market design + econometrics; PhD is the stated qualification. Seattle-commutable (SF/Palo Alto/Seattle, 1–2x/month in-office). Comp: IC15 $462K total ($240K base); IC16 $704K total ($271K base).

**JD Key Requirements (Sr. MLE/Economist, Ads Marketplace, verified):**
- MS or PhD in CS, Economics, Statistics, or related field (required at Senior level)
- "Knowledge in auction theory, market design, and econometrics"
- "Industry experience applying economics or ML to ads auctions, pricing, marketplaces"
- A/B testing, causal inference, online experimentation at large scale
- "Implement auctions, tune marketplace parameters (utility function), model long-term effects to reduce ad fatigue"
- Focus areas: ranking/pricing/mechanism design, bidding/budgeting innovation, advertiser churn/retention modeling, second-order effects for new ad offerings

**Interview Process (from Glassdoor/Blind/Exponent/InterviewQuery, 2024–2025):**
1. Recruiter screen (30 min)
2. Phone screen (45–70 min) — 3–5 ML conceptual questions + 1–2 LeetCode medium/hard coding problems (Python)
3. Onsite (4–6 rounds, 1–4 weeks total):
   - 2 coding rounds (LeetCode medium with hard follow-ups; topics: graphs, trees, sliding window, backtracking, matrix operations; framed in Pinterest context)
   - 1 ML concepts round
   - 1–2 ML system design rounds (documented topics: ads ranking algorithm, product recommendation, spam detection, visual search, advertiser recommendation)
   - 1 behavioral round (evaluated against 5 core values — see insider notes)
- Total timeline: 1–4 weeks typical; staff-level processes can extend to 2 months with 7 rounds
- Interviewers explain scoring rubric upfront

**Key insider notes:**
- Five core values evaluated across ALL rounds (not just behavioral): **Put Pinners First** (user-centric decisions), **Aim for Extraordinary** (high execution bar), **Create Belonging** (collaborative, inclusive AI), **Act as One** (team goals > individual), **Win or Learn** (calculated risk-taking)
- Inclusive AI / algorithmic fairness in ranking is explicitly evaluated — expect to be asked how you'd handle fairness trade-offs in ads ranking
- Staff-level bar (IC16) significantly higher than Senior (IC15) — scope-of-impact stories must be at org level, not team level
- Recruiter communication issues at staff level: documented cases of >1 week silence after positive screens — follow up proactively at 5–7 business days
- Read these blog posts before interviews: "Beyond Two Towers: Re-architecting the Serving Stack" (Feb 2026), "Evolution of Ads Conversion Optimization Models", "Handling Online-Offline Discrepancy in Pinterest Ads Ranking"
- Stack: PyTorch/TF, MLFlow, Spark, Flink, Kafka, Kubernetes, Ray; **Galaxy** (homegrown online feature store), **Scorpion** (feature fetching + model inference platform), **MLEnv** (full-stack ML developer framework); ads ranking architecture evolved: GBDT+LR (2018) → DNN+MTL (2020) → AutoML multi-tower (2021) → MMOE-DCN (2024) → GPU serving (2025)

---

### Whatnot

**Titles:** Machine Learning Scientist, Software Engineer ML

| Role | Level | URL |
|---|---|---|
| Machine Learning Scientist | Senior (4–5+ yrs) | jobs.whatnot.com (Ashby) — search "Machine Learning" |
| Software Engineer, Machine Learning | Senior | jobs.whatnot.com — search "Machine Learning" |

**Best fit for you:** Machine Learning Scientist — real-time ranking, auction discoverability, seller trust signals. Your auction/pricing background maps directly onto Whatnot's core ML problems.

**JD Key Requirements (ML Scientist, from YC/Built In postings, 2025):**
- 4–5+ years ML in production; end-to-end model ownership (data → training → serving → monitoring)
- Python (primary), SQL; PyTorch, LightGBM/scikit-learn
- Experience with real-time ML systems and low-latency serving
- Marketplace, ranking, or recommendation systems experience preferred
- Fraud/trust & safety ML background valued

**Interview Process (from Glassdoor, TeamBlind, 1Point3Acres, 2024–2025):**
1. OA — Karat-style coding assessment (LeetCode medium difficulty)
2. Technical screen — coding round (data structures + algorithms)
3. Virtual onsite (4–5 rounds):
   - **Coding** (~60 min) — LeetCode medium–hard; trie, graphs, DP documented; working solution first
   - **System design** (~60 min) — standard scalable system; real-time ranking, fraud pipeline, auction price predictor are relevant topics
   - **Product sense** (~60 min) — **unique to Whatnot**: must have used the app; "what would you add/improve?"; tie ML suggestions to business metrics (GMV, seller retention, engagement)
   - **Hiring manager** (~30 min) — past work, impact, leadership signal
   - **Values/principles** (~30 min) — low ego, growth mindset, high-impact drive, community first
- Timeline: ~26 days average

**Key insider notes:**
- **Use the app before interviews** — product sense round is a real gate; create a seller and buyer account, attend live auctions, note UX gaps and ML improvement opportunities (bidder fraud signals, show discoverability, price prediction confidence intervals)
- Coding bar is LeetCode medium — not FAANG hard; working solution + clean code matters more than exotic optimization
- System design: Whatnot's documented ML stack is GBDT → compiled C++ binaries (<200ms p99 via Rockset + Redis); name this framing if asked about inference optimization
- Trust & safety is a first-class ML problem: LLM-enhanced multimodal moderation, fraudulent bidding detection, seller reputation scoring — all documented and interview-relevant
- Values are evaluated across all rounds, not just the principles round — frame every design decision in terms of seller/buyer community impact
- Stack: Kafka (Confluent Cloud), KSQL, Snowflake, Rockset, Redis, PyTorch, LightGBM, gRPC, FastAPI
- Read before interviews: "Evolving Feed Ranking at Whatnot", "6x Faster ML Inference: Why Online > Batch", "Feeds with Real-time Signals", "How Whatnot Utilizes Generative AI for Trust & Safety" — all on medium.com/whatnot-engineering

### Microsoft

**Target role:** Principal Applied Scientist, Auction & Bidding *(added July 2026)*

| Role | Level | URL |
|---|---|---|
| Principal Applied Scientist, Auction & Bidding | L64 (Principal) | [Microsoft Careers](https://apply.careers.microsoft.com/careers/job/1970393556852883) |

**Best fit for you:** Direct overlap with DoorDash Ads + Amazon Sponsored Products background — auto-bidding, auction mechanism design, advertiser ROI optimization. 3+ years auto-bidding required (you have it); PhD preferred.

**JD Key Requirements (verified July 2026):**
- BS+6yr / MS+4yr / PhD+3yr in Statistics, Econometrics, CS, or quantitative field
- 8+ years developing and deploying production systems within product teams
- 3+ years auto-bidding or auction design
- Design + analyze bidding strategies using optimization and control theory
- Prototype auction mechanisms for specific product areas; large-scale A/B experimentation for marketplace health
- Build automation algorithms to improve advertiser ROI via AI/ML
- Causal reasoning models (explicitly listed in JD)

**Interview Process (from Glassdoor + Blind + InterviewQuery, 2025–2026):**
1. Recruiter screen (30 min) — background, comp alignment, timeline
2. Technical screen with HM or Senior Scientist (45–60 min) — resume deep-dive + 1 domain-relevant problem (auto-bidding or optimization framing)
3. Virtual onsite loop (4–5 rounds, same day):
   - **Coding** — DS&A, LeetCode medium/hard (2 problems, ~45 min); clean code + complexity analysis required
   - **ML depth** — model design, feature engineering, experimentation; first-principles derivations expected at L64+
   - **ML system design** — design an auto-bidding system or auction serving pipeline; expect scale, latency, and monitoring discussion
   - **Domain/Research depth** — auction theory, control theory for bidding, causal inference for incrementality; derive GSP equilibrium or constrained optimization setup from scratch
   - **Behavioral/Culture** — Microsoft "growth mindset" values; STAR format; skip-level manager may attend
4. Research talk (L64+): 20–30 min presentation of past work → Q&A on methodology, production impact, limitations
- Total: ~3–5 weeks average

**Key insider notes:**
- Auto-bidding depth is the hiring signal differentiator — derive target-CPA/target-ROAS optimization (Lagrangian relaxation, dual decomposition) from first principles; do not just name-drop
- Auction mechanism questions: GSP vs VCG trade-offs, first-price auction equilibrium, revenue equivalence theorem — expect at least one derivation
- Control theory framing: PID controllers for bid adjustment / budget pacing; LQR for multi-constraint optimization — know the vocabulary, be able to position alternatives
- Causal inference is a stated JD requirement: incrementality measurement via geo-holdout, synthetic control, or DID — design an experiment end-to-end
- Microsoft Advertising (MSAI) team runs Bing Ads / Microsoft Advertising platform — frame all past work in terms of advertiser ROI and marketplace health, not just model metrics
- L64 principal comp: ~$280K–$420K+ TC (Redmond / Mountain View); no forced ranking curve (different from Meta stack-rank culture)
- Growth Mindset culture: prepare a concrete story about learning from failure or reversing a major technical decision under pressure
- Reports on Blind flag an intense L62 loop — L64 principal bar adds research talk and domain depth; plan for a 5–6 hour onsite day

---

## 1. The Overlap Strategy

### Coding — Universal Core (applies to all 10)

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

**Non-negotiables across all 10:**
- Clean code, typed signatures, edge case narration before coding
- Complexity analysis stated before optimization pass
- Think aloud — all 10 companies score communication as a dimension

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

**Core system designs to master (used by 6+ of 10 companies):**
- Two-tower retrieval + re-ranking pipeline (Meta, Netflix, Roblox, Uber, Pinterest)
- Ads auction + bidding system end-to-end (Meta, Uber, Reddit, Pinterest)
- Real-time feature serving with low-latency SLA (all 10)
- A/B testing + experimentation platform design (all 10)

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

Your background is a genuine moat for ~8 of the 10 companies. Here's how to deploy it surgically.

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

**Reddit — Autobidding Platform in Transition**
- Reddit is moving from simple CPC → full CPA/ROAS autobidding; this exact transition is the core problem you solved at DoorDash.
- Your angle: "Reddit's autobidding system is at the stage where you have to rearchitect pacing from impression-based to outcome-based signals. The hard part is handling the cold-start exploration/exploitation trade-off during the learning period, and ensuring the feedback loop doesn't destabilize when campaign sizes are small — I've shipped exactly this at DoorDash."
- Mechanism design hook: bid shading in first-price auction environments, convergence properties of autobidding when multiple advertisers simultaneously run learning algorithms, advertiser learning period tradeoffs (short exploration hurts ROI, long exploration wastes budget)
- Stack angle: Reddit uses control theory + RL for pacing (different framing from pure Lagrangian) — bridge this: "Lagrangian relaxation and PID control are solving the same budget constraint problem — they differ in whether you update the dual variable in closed form or via gradient; Reddit's RL framing naturally handles non-stationarity."

**Pinterest — Constrained Organic/Paid Auction Design**
- Pinterest's structural challenge: auction over visual discovery content where promoted pins must not degrade the organic discovery experience users come for.
- Your angle: "This is a mechanism design problem with a quality constraint — maximize advertiser value subject to a user experience floor. The same multi-objective structure appears in every ads marketplace, but Pinterest's visual/discovery context makes the quality signal non-trivial: a promoted pin that 'looks organic' may convert better short-term but creates trust erosion over time. I'd design the auction with an explicit quality multiplier that incorporates long-horizon engagement signals, not just immediate CTR."
- Mechanism design hook: constrained revenue maximization (Lagrangian dual of quality floor constraint), long-term advertiser retention vs. short-term revenue, second-order effects of ad load on organic engagement, ad fatigue modeling
- Stack angle: Pinterest's MMOE-DCN architecture separates user tower from item tower — in a mechanism design context, the user tower is estimating the user's utility function, and the item tower is the advertiser's bid; the auction mechanism combines them. Understanding this framing will differentiate your system design answer.

**Whatnot — Livestream Auction Mechanics & Seller Marketplace**
- Whatnot's core ML challenge: real-time ranking and price discovery in livestream auctions where bidder intent, item rarity, and seller reputation are all noisy, fast-moving signals.
- Your angle: "Whatnot's auction discoverability problem is a mechanism design problem under information asymmetry — collectors have private valuations, sellers have private quality information, and the platform needs to surface the right shows to the right bidders without creating adverse selection. My DoorDash Ads experience designing bid landscapes and quality score systems maps directly."
- Mechanism design hook: adverse selection in collectibles auctions (fake grading, misrepresented condition), reputation cascades (high-trust sellers always win visibility), information revelation via bidding patterns.
- Product sense preparation: frame ML improvements as seller GMV + buyer trust metrics — not abstract model accuracy.

### The 30-Second Pitch Template (use in intros)
> "I'm a Senior MLE at DoorDash focused on ads economics — specifically auction mechanism design, autobidding, and budget pacing. My PhD is in computational game theory, so I approach marketplace ML problems by first modeling the strategic behavior of agents — advertisers, platforms, users — before choosing an ML approach. That lens is particularly useful for systems where the model's outputs affect future inputs, which is true in any closed-loop bidding or pricing system."

---

## 3. Company-Specific Deep Dives (The Deltas)

### Cluster A: Ads/Marketplace ML — Meta, Uber, Reddit, Pinterest, Whatnot, Microsoft
**Common thread:** Real-time bidding, two-sided marketplace, high-throughput feature serving, auction mechanism design
**Your delta:** Minimal — this is your core domain
**What to add:**
- Meta: DLRM architecture, FAISS/ScaNN for embedding retrieval, Flink feature pipelines
- Uber: H3 geospatial indexing, Michelangelo feature store, prediction intervals for ETA (not just point estimates)
- Reddit: Control theory + RL framing for budget pacing (vs. Lagrangian); Kafka/Flink stack for 65B+ daily events; phone screen is practical ML not LeetCode — practice building a working model in 45 min
- Pinterest: Galaxy (homegrown feature store) + Scorpion (inference platform) + MLEnv (ML framework) — name these in system design; MMOE-DCN two-tower ads ranking architecture; read "Beyond Two Towers" (Feb 2026) before interviews; fairness/inclusive AI is explicitly evaluated
- Whatnot: GBDT-based feed ranking + online inference (<200ms p99 via Rockset + Redis); seller trust scoring (fulfillment %, cancellation %, review signals); **product sense round requires app usage** — prepare 3 ML-improvement pitches tied to GMV/engagement metrics; read Whatnot Engineering Medium blog before interviews

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
**Ads context (verified July 2026):** Target role is now **Research Scientist 5, Marketplace - Ads DSE** — owns ML and optimization algorithms for ad quality, targeting, ranking, and bidding on the Ad Marketplace team. RS5 ≈ Staff bar. This is a production-accountability research role, not pure research — deployed impact is required.

**Your angle:** "Netflix Ads is building the optimization layer on top of an already-scaled delivery platform — the gap is the ML rigor on targeting, pacing, and auction design. My work designing those systems from first principles at Amazon and DoorDash maps directly, and I can speak to the same subscriber-retention tension (ads must not degrade watch experience) from the marketplace health angle." Lead with the research depth (mechanism design PhD), close with production credibility.

**Preparation:**
- Ad inventory forecasting: predicting available impression supply by targeting segment — time-series + uncertainty quantification (prepare this as a system design)
- Ad pacing under inventory constraints: Lagrangian relaxation framing (same as budget pacing at DoorDash)
- Ad quality optimization: multi-objective ranking (relevance, pacing, revenue) — this is the core RS5 problem
- Research deep-dive prep: select one project to present end-to-end (problem → methodology → tradeoffs → deployed impact → what you'd do differently)
- Know their ML stack: Metaflow-style DAG orchestration, real-time inference SLA requirements
- Product-mindedness required: tie every design decision to subscriber retention — this tension is unique to Netflix vs. pure-play ad platforms

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
| 4 | Ads/Marketplace (Meta, Uber, Reddit, Pinterest, Whatnot) | Rate limiters, task schedulers, online algorithms | Autobidding system + budget pacing + Pinterest ads ranking + Whatnot seller trust scoring | Impact story from DoorDash Ads |
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
| "Budget Pacing for Targeted Online Advertisements at LinkedIn" (KDD 2014) | Meta, Reddit | Your domain — know this deeply |
| "Autobidding with Constraints" (Google, 2021) | Meta, Reddit, Pinterest | Direct relevance to your work |

### Engineering Blogs (Bookmark and Read Last 12 Months)

| Blog | Key Posts to Find |
|---|---|
| Meta AI / Meta Engineering | Ads ranking, DLRM, recommendation systems |
| Uber Engineering (eng.uber.com) | ETA improvements, Michelangelo, surge pricing |
| Netflix Tech Blog (netflixtechblog.com) | Recommendation, Metaflow, A/B testing |
| Stripe Engineering (stripe.com/blog/engineering) | Fraud detection, ML at Stripe, reliability |
| Shopify Engineering (shopify.engineering) | ML for e-commerce, infrastructure |
| OpenAI Research (openai.com/research) | GPT-4 system card, alignment updates |
| Pinterest Engineering (medium.com/pinterest-engineering) | "Beyond Two Towers" (Feb 2026), "Evolution of Ads Conversion Models", "Handling Online-Offline Discrepancy in Ads Ranking", "MMOE-DCN Architecture", MLEnv platform |
| Reddit Engineering (redditinc.com/blog / reddit.com/r/RedditEng) | Ads delivery, ranking systems, safety ML, Kafka/Flink data infra |
| Google Research (on auctions) | First-price auction transition, bid shading |
| Whatnot Engineering (medium.com/whatnot-engineering) | "Evolving Feed Ranking at Whatnot", "6x Faster ML Inference: Why Online > Batch", "Feeds with Real-time Signals", "How Whatnot Utilizes Generative AI for Trust & Safety", "Whatamix: Blendable Feed Construction" |

### Frameworks & Tools to Know by Name (not deep expertise required)

| Tool | Context |
|---|---|
| FAISS / ScaNN | Approximate nearest neighbor for embedding retrieval (Meta, Netflix) |
| Flink / Spark | Streaming vs. batch feature pipelines |
| Ray / Metaflow | ML workflow orchestration (Netflix, Uber) |
| Triton Inference Server | GPU-optimized model serving (OpenAI adjacent) |
| Feature stores: Feast, Tecton, Hopsworks | Know the online/offline consistency problem |
| Galaxy + Scorpion (Pinterest) | Pinterest's homegrown online feature store + inference platform — name these in Pinterest system design |
| MLEnv (Pinterest) | Pinterest's full-stack ML developer framework — abstracts dataset loading, model arch, training loop |
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
- [ ] Write 7 system design docs (auction, two-tower, feature store, recommendation, A/B platform, LLM serving, seller trust scoring)
- [ ] Read 5+ engineering blog posts per target company
- [ ] Complete 4+ mock interviews (2 coding, 1 system design, 1 behavioral)
- [ ] Prepare 2–3 company-specific questions per company showing blog/paper familiarity

---

## Appendix: Other Companies Evaluated (Research as of March 2026)

**Snap** — Principal MLE, Ads Marketplace (Palo Alto only, ~$480K median total). Strong RTB auction + DL ads ranking match; Palo Alto HQ only — no remote option confirmed. Monitor if open to relocation. Interview: no dedicated behavioral round; values ("Kind, Smart, Creative") evaluated across all rounds; team matching after offer can take weeks.

**Skipped:** Expedia ($184K–$258K, below bar), Lyft ($176K–$220K + hybrid 3 days/week), Amazon DSP (prior employer — evaluate separately), Google/YouTube Seattle (3–6 month process, LeetCode-heaviest bar), TikTok/ByteDance (geopolitical risk).
