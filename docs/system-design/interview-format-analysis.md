# System Design Interview Format by Company

**Purpose:** Clarify whether each target company requires ML system design, traditional system design, or both — and derive prep prioritization accordingly.
**Coverage:** 10 target companies
**Source:** `../prep-roadmap.md` (Glassdoor, InterviewQuery, Blind, Taro — verified March 2026)

---

## 0. TL;DR — Quick Reference

| Company | ML System Design | Traditional System Design | SD Round Count | Sample Questions |
|---|---|---|---|---|
| **Meta** | ✅ yes | ❌ no | 1 | Feed ranking, ads CTR prediction, recommendation system |
| **Netflix** | ✅ yes | ❌ no | 1 | Recommendation architecture, ad-break prediction, online training pipeline |
| **Uber** | ✅ yes | ❌ no | 1 | Real-time inference architecture, failure modes, observability |
| **Shopify** | ✅ yes | ❌ no | 1 | Real-time ML serving (<100ms), Black Friday resilience, feature pipeline (Pano/Feast) |
| **OpenAI** | ✅ yes | ❌ no | 1 | Unsafe content detection, scalable recommender, topic classification at scale |
| **Stripe** | ✅ yes | ❌ no | 1 | Fraud detection, merchant recommendations, authorization optimization |
| **Pinterest** | ✅ yes | ❌ no | 1–2 | Ads ranking, product recommendation, spam detection, visual search |
| **Roblox** | ✅ yes | ❌ no | 1 | Recommendation system (documented as anchor question) |
| **Reddit** | ✅ yes | ✅ **yes** | **2** | ML: stock prediction from comments, recommendation; Generic: standard distributed systems |
| **Whatnot** | ⚠️ hybrid | ⚠️ hybrid | 1 | "Standard scalable system" prompt but topics are ML-adjacent (ranking, fraud pipeline, auction price predictor) |

**Bottom line:**
- **10/10** companies have ML system design — it is the universal requirement
- **1/10** (Reddit) explicitly has a separate **generic system design** round in addition to ML SD
- **1/10** (Whatnot) has an ambiguous hybrid round — generic SD framing with ML-adjacent content
- **8/10** are ML system design only — traditional distributed systems design is not tested

---

## 1. What's the Difference?

#### ML System Design
Covers the end-to-end ML lifecycle applied to a product problem.

| Phase | What's evaluated |
|---|---|
| Problem framing | Metrics, objective alignment, what to predict |
| Data modeling | Features, labels, storage, online/offline split |
| ML pipeline | Training → serving → monitoring |
| Failure modes | Fallbacks, degraded mode, circuit breakers |
| Capacity | QPS, latency budget, storage, compute estimates |

**Signal sought:** Can you build and operate an ML system at production scale? Do you think end-to-end — not just about the model?

---

#### Traditional (Generic) System Design
Covers distributed systems architecture for scalable infrastructure.

| Phase | What's evaluated |
|---|---|
| Requirements | Scale (QPS, storage), consistency, availability |
| API design | REST/gRPC contracts, versioning |
| Storage | Database choice, schema, sharding, indexing |
| Caching | CDN, Redis, invalidation strategy |
| Scalability | Horizontal scaling, load balancing, partitioning |
| Reliability | Replication, failover, CAP theorem trade-offs |

**Signal sought:** Can you design a horizontally scalable distributed system? Do you understand consistency models and failure boundaries?

---

#### Why the distinction matters for prep
ML system design and traditional system design test overlapping but distinct knowledge:

| Dimension | ML System Design | Traditional System Design |
|---|---|---|
| Primary knowledge domain | ML algorithms, feature engineering, model serving, monitoring | Distributed systems, consensus, storage engines, networking |
| Storage emphasis | Feature stores, model registries, training data pipelines | Relational/NoSQL schemas, sharding strategies, replication |
| Latency framing | Inference latency budget, model size vs. QPS tradeoff | API p99, database read/write latency, CDN offload |
| Scale framing | Training compute, feature freshness, batch vs. real-time | QPS, storage growth, partition count |
| Failure mode emphasis | Model staleness, drift, fallback ranking | Split-brain, data loss, cascading failure |
| Unique requirements | Monitoring/drift, A/B holdout, shadow scoring | CAP theorem trade-offs, consistent hashing, write-ahead log |

Preparing only for ML system design leaves a gap on Redis internals, consistent hashing, B-tree vs. LSM-tree storage, and distributed consensus — topics that appear in Reddit's generic round and are implied in Whatnot's hybrid round.

---

## 2. Per-Company Breakdown

---

### Meta — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1 ML system design round in the onsite (of 4–6 total rounds)
- No generic system design round documented

**Documented topics:**
- News Feed ranking end-to-end
- Ads CTR prediction
- Recommendation system design
- "Design an evaluation framework for ads ranking" (documented question)

**What Meta probes:**
- Scale: 1B+ users; address sharding, hot partitions, geographic distribution
- DLRM familiarity; two-tower models; ANN via FAISS / ScaNN
- p99 online serving <100ms; Flink/Spark feature pipelines
- PyTorch + FSDP/DDP; gradient checkpointing; mixed precision

**Prep priority:** ML SD only. No traditional SD prep needed.

---

### Netflix — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1 ML system design round in the virtual onsite (of 3 total)
- No generic system design round documented

**Documented topics:**
- Online training pipelines
- Ad-break prediction (cited as actual question)
- Recommendation architecture
- Latency budgets (<200ms) explicitly required

**What Netflix probes:**
- Product thinking: every design decision tied to a user or business metric
- Monitoring upfront: drift detection, shadow scoring, A/B holdout stated before they ask
- Real-time inference: p99 SLA under sustained load; explicit fallback chain
- Metaflow-style DAG awareness; feature store online/offline consistency

**Prep priority:** ML SD only. Monitoring and fallback chain are the highest-signal differentiators.

---

### Uber — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- Round 3 of 4–5 onsite rounds is dedicated ML System Design
- No generic system design round documented

**Documented topics:**
- Real-time inference architecture
- Failure modes and observability
- Marketplace cascade effects (how prediction errors flow through pricing/dispatch)

**What Uber probes:**
- Geospatial ML: H3 hexagonal indexing, spatial feature engineering, surge zone modeling
- ETA modeling: prediction intervals required — point estimates alone fail
- Michelangelo conventions; feature store reuse across ETA/Pricing/Matching
- Sub-second feature freshness; driver supply/demand signal latency

**Prep priority:** ML SD only. Geospatial features and prediction intervals are differentiating at L6.

---

### Shopify — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- ML system design is the primary technical design round in onsite
- No generic system design round documented

**Documented topics:**
- Real-time ML serving (<100ms inference)
- Feature pipeline design via Pano (Shopify's Feast-based feature store) and Kafka
- Black Friday resilience: 10× traffic spikes, failover, degraded-mode serving
- Multi-region deployment, A/B test design for GMV/conversion

**What Shopify probes:**
- SOLID principles enforced at class level; flag God objects and leaky abstractions
- Trade-off framing: ≥2 design options with explicit reasoning before committing
- Testability: dependencies injected, not hardcoded
- Name Pano + Ray + Kafka explicitly — they probe platform familiarity

**Prep priority:** ML SD only. Black Friday resilience and explicit decision logging are unique Shopify signals.

---

### OpenAI — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1 ML system design round in the virtual onsite (of 4–6 rounds over 1–2 days)
- No generic system design round documented

**Documented topics:**
- Unsafe content detection pipeline with bias monitoring
- Scalable recommender system with distributed computing
- Topic classification at scale
- Safety systems

**What OpenAI probes:**
- Production debugging: exploding gradients, vanishing signals, batch norm instability
- AI ethics discussion is real — positions on RLHF trade-offs, reward hacking, dual-use risks
- Mission alignment: intellectual honesty over polish
- Bias monitoring in the system design (not as afterthought)

**Prep priority:** ML SD only. Bias monitoring and safety systems are OpenAI-specific additions to the standard ML SD rubric.

---

### Stripe — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1 ML system design round in the onsite (of 4 rounds)
- No generic system design round documented (Stripe has a unique Debugging round instead)

**Documented topics:**
- Fraud detection architecture
- Merchant recommendations
- Authorization optimization

**What Stripe probes:**
- Fraud detection: adversarial dynamics (fraud patterns shift when detection improves)
- Real-time vs. batch trade-offs for transaction scoring (<100ms at checkout)
- No LeetCode mentality — production code and real-world failure modes expected

**Prep priority:** ML SD only. Fraud detection is a domain-specific lens on standard ML SD.

---

### Pinterest — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1–2 ML system design rounds in the onsite (staff-level gets 2)
- No generic system design round documented

**Documented topics:**
- Ads ranking algorithm
- Product recommendation
- Spam detection
- Visual search
- Advertiser recommendation

**What Pinterest probes:**
- Inclusive AI / algorithmic fairness in ranking — expect fairness trade-off questions
- Galaxy (homegrown online feature store), Scorpion (feature fetching + inference), MLEnv
- MMOE-DCN architecture familiarity for ads ranking
- Multi-objective optimization: relevance + diversity + fairness simultaneously

**Prep priority:** ML SD only. Fairness and multi-objective optimization are Pinterest-specific differentiators.

---

### Roblox — ML System Design only

**Classification:** ML SD ✅ | Traditional SD ❌

**Round structure:**
- 1 ML system design round in onsite (of 4–5 rounds)
- No generic system design round documented

**Documented topics:**
- Recommendation system design (documented as anchor question)
- Safety/moderation systems (secondary)

**What Roblox probes:**
- Recommendation system end-to-end (same template as Netflix — two-tower + ranking)
- Platform economics / ads framing emerging but not yet primary
- Process described as more structured and friendly than OpenAI/Netflix

**Prep priority:** ML SD only. Recommendation system is the most likely prompt — prepare the Netflix worked example as direct practice.

---

### Reddit — ML System Design **AND** Traditional System Design

**Classification:** ML SD ✅ | Traditional SD ✅ | **Both confirmed**

**Round structure (6-hour virtual onsite):**
1. ML system design round — domain-specific to Reddit
2. **Generic system design round** — standard distributed systems (explicitly listed as separate round)
3. Pair programming / coding round
4. Behavioral + cross-functional round
5. Hiring manager round

**Documented ML SD topics:**
- Stock prediction from Reddit comments
- Recommendation systems
- Content ranking, spam detection

**Generic SD topics (inferred from "standard" designation):**
- Likely: design a feed (timeline ranking), URL shortener, notification system, search index
- May blend into ML territory given Reddit's product (content ranking is both SD and ML SD)

**What Reddit probes:**
- ML SD: practical, domain-grounded — problems feel like actual Reddit product challenges
- Generic SD: distributed systems fundamentals — consistent hashing, database sharding, message queues, caching
- Phone screen is live ML model building (45 min) — not LeetCode; build working model on real dataset

**Prep priority:** **Both ML SD and Traditional SD.** Reddit is the only confirmed company requiring generic system design prep.

---

### Whatnot — Hybrid (ambiguous)

**Classification:** ML SD ⚠️ hybrid | Traditional SD ⚠️ hybrid

**Round structure:**
- 1 system design round (~60 min)
- Framing: "standard scalable system" — traditional SD prompt style
- Content: "real-time ranking, fraud pipeline, auction price predictor relevant" — all ML-adjacent

**Interpretation:**
- The prompt is likely a traditional SD-style question (e.g., "design a real-time bidding system") where the candidate is expected to cover both infrastructure scalability AND ML components
- Whatnot's documented ML stack (GBDT → compiled C++ binaries, Rockset + Redis, <200ms p99) suggests they want infrastructure depth alongside model design
- Not a pure ML SD (no training pipeline deep-dive expected) and not pure traditional SD (ML inference is central)

**What Whatnot probes:**
- Scalable real-time infrastructure: Kafka, KSQL, Redis, event streaming
- ML-adjacent: ranking model serving, fraud signal pipelines, auction price confidence
- Trust & safety: LLM-enhanced multimodal moderation, fraudulent bidding detection
- Their stack: Confluent Kafka, KSQL, Snowflake, Rockset, Redis, PyTorch, LightGBM

**Prep priority:** Cover distributed systems infrastructure primitives (message queues, stream processing, caching) at a working level. Full traditional SD deep-dive not required — Whatnot topics are narrow enough to prep by covering their specific stack.

---

## 3. Prep Strategy Implications

#### Time Allocation

| Area | Allocation | Covers |
|---|---|---|
| **ML System Design** | **80%** of system design prep | All 10 companies |
| **Traditional System Design** | **15%** | Reddit (confirmed), Whatnot (hedge) |
| **Hybrid / Infrastructure depth** | **5%** | Whatnot-specific stack (Kafka, Rockset, Redis internals) |

#### Priority ordering
1. **ML SD first** — 10/10 companies; the template + worked examples in this folder are the primary prep artifact
2. **Reddit traditional SD** — only company requiring full generic distributed systems prep; treat as a separate prep track
3. **Whatnot infrastructure hedge** — understand Kafka, stream processing, and Redis at the operational level (not just feature store usage)

#### What changes by company
| If your next loop is... | Prep focus |
|---|---|
| Meta / Netflix / Uber | ML SD deep-dive only; company-specific rubric (scale mechanisms, monitoring, ETA intervals) |
| Shopify | ML SD + Black Friday resilience pattern; name Pano explicitly |
| OpenAI | ML SD + safety/bias monitoring + production debugging |
| Pinterest | ML SD + fairness/multi-objective trade-offs; read Galaxy/Scorpion platform papers |
| Stripe | ML SD through fraud lens; adversarial dynamics + real-time transaction scoring |
| Roblox | ML SD for recommendation system; use Netflix worked example as direct prep |
| **Reddit** | **Both ML SD and traditional SD**; prep generic distributed systems in parallel |
| **Whatnot** | **Hybrid**; cover their specific stack; prepare for infrastructure + ML serving in same design |

---

## 4. Traditional SD Quick Reference (for Reddit + Whatnot)

> Only needs to be covered at working depth — not at the same level as ML SD. Focus: enough to handle 45–60 min with a competent Reddit/Whatnot interviewer.

#### Core Distributed Systems Primitives to Know

| Primitive | What to know |
|---|---|
| Consistent hashing | Ring-based sharding; virtual nodes; how to add/remove nodes without full rehash |
| LSM-tree vs. B-tree storage | LSM (Cassandra, RocksDB): write-optimized, compaction cost; B-tree (Postgres): read-optimized, write amplification |
| Write-ahead log (WAL) | Durability guarantee; used by Postgres, Kafka (commit log), Redis AOF |
| CAP theorem in practice | CP vs. AP; why most distributed DBs choose AP + eventual consistency; when you need CP |
| Kafka internals | Topic → partition → offset; consumer groups; at-least-once vs. exactly-once delivery |
| Rate limiting | Token bucket vs. leaky bucket; sliding window counters; Redis-based distributed rate limiter |
| Database sharding | Horizontal (by key range or hash); hot partition problem; cross-shard query cost |
| Caching patterns | Cache-aside vs. write-through vs. write-behind; TTL; cache stampede / thundering herd |
| CDN | Edge caching; cache invalidation; origin shield pattern |
| Replication | Leader-follower; quorum reads/writes; split-brain and fencing |

#### Reddit-Specific Topics
- **Feed design** — timeline ranking at scale: fan-out on write vs. fan-out on read; hot users problem
- **Content spam pipeline** — heuristic pre-filter → ML scoring → human review queue; same two-stage pattern as ML SD
- **Search index** — inverted index, Elasticsearch sharding, real-time vs. near-real-time indexing
- **Notification system** — push vs. pull, delivery guarantees, preference management at scale

#### Whatnot-Specific Topics
- **Real-time auction infrastructure** — bid ingestion (Kafka), state management (Redis), leaderboard (sorted sets), settlement
- **Live video + chat scale** — WebSocket connection management, message fanout, presence detection
- **Fraud detection pipeline** — real-time feature computation (KSQL), scoring (LightGBM), action enforcement (<200ms)
- **Rockset** — real-time analytics DB; Whatnot uses it for online feature serving alongside Redis; know its convergent index architecture

---

## 5. Open Questions to Clarify Before Reddit/Whatnot Loops

- [ ] **Reddit generic SD** — ask recruiter whether the generic SD round is infrastructure-focused or product-focused. If they say "design Reddit's feed", that blends into ML SD territory
- [ ] **Whatnot SD framing** — confirm with recruiter: "Is the system design round focused on infrastructure scalability or ML system design?" The answer determines whether to prep traditional SD primitives at depth
- [ ] **Pinterest 1 vs. 2 SD rounds** — documented as 1–2 at staff level; confirm count early; second round is typically a different domain (e.g., ads ranking + visual search)
