# Roblox Game/Experience Recommendation — ML System Design

**Domain:** `ranking`
**Target Company:** Roblox
**Difficulty Bar:** L6 (Staff MLE)
**Date:** 2026-03-27
**Related Designs:** `netflix-recommendation-system.md`, `../templates/ml-system-design-template.md`

---

## 0. Scorecard

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★★★★★ | — |
| Data Modeling | ★★★★☆ | Cross-age-group content filtering complexity understated |
| ML Pipeline | ★★★★★ | — |
| Failure Modes | ★★★★☆ | Creator ecosystem health guardrail metric partially specified |
| Capacity | ★★★★★ | — |

**Overall:** `STRONG HIRE`
**Top Gap:** Age-appropriate content filtering — Roblox serves ages 7–25+; content rated for 17+ must be hard-blocked for under-13 users regardless of recommendation model output.

---

## 1. Requirements

#### Functional Requirements
1. Recommend Roblox experiences (games) to players on the Home page, Discover tab, and search results
2. Handle cold-start for new players (no play history) and new games (no engagement data)
3. Incorporate social signals: surface games that friends are actively playing
4. Enforce age-appropriate content filtering: hard-block experiences with maturity ratings above user's age group
5. Support creator ecosystem health: prevent top-10 games from monopolizing recommendations

#### Non-Functional Requirements

| Dimension | Target | Business Tie |
|---|---|---|
| Latency (p99 serving) | ≤ 150ms | Home page and Discover tab load time; impacts DAU retention |
| Availability | 99.99% | Recommendation unavailability = players see generic charts; engagement drops |
| Consistency | Eventual (minutes) | Session signals update within 5 min; friend-activity signals within 30s |
| Throughput | ~120K peak QPS | 80M DAU × ~10 recommendation requests/day / 86,400 × 2× peak |
| Feature freshness (friend activity) | ≤ 30s | Friends' live game presence is a strong real-time signal |
| Creator equity | Bottom 50% of creators receive ≥ 15% of recommendation slots | Guards against monopolization by top titles |

#### Scale Numbers (stated upfront)
- **DAU / MAU:** 80M DAU / 200M+ MAU
- **Active experiences:** ~40M user-created games (vast majority have < 100 players)
- **Peak concurrent players:** ~4M simultaneously
- **Peak QPS:** ~120K
- **Player age distribution:** ~50% under 18; ~25% under 13 (COPPA-regulated)

#### Out of Scope
- In-game content moderation (separate safety system)
- Virtual economy (Robux transactions, avatar items)
- Creator monetization and revenue share
- Search ranking (separate system; different recall requirements)

> **Roblox rubric:** Safety layer is a first-class design requirement — not a compliance footnote. Age-appropriate filtering must be a **hard filter** before ranking, not a soft feature score. Creator ecosystem health is a business objective, not just a nice-to-have.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| Player play history embedding | `batch` | 1 hour | Spark on session logs → feature store | Mean-pool of experience embeddings for last 200 sessions |
| Session-recent plays (last 5 experiences) | `real-time` | ≤ 5 min | Kafka → Flink → Redis | Current session intent; strongest short-term signal |
| Friend activity (games friends are playing now) | `real-time` | ≤ 30s | Presence service → Kafka → Redis | "3 friends are playing X" dramatically boosts click-through |
| Friend play history overlap | `batch` | daily | Spark on friend graph × play logs | Collaborative filtering signal scoped to friend graph |
| Experience content embedding | `batch` | daily | NLP on game description + genre tags + screenshot CLIP | 256-d; enables cold-start for new experiences |
| Experience genre tags | `static` | at publish | Creator catalog | Multi-label: ["FPS", "Roleplay", "Obby", "Simulator"] |
| Experience popularity (DAU, concurrent players) | `batch` | 1 hour | Spark aggregation on session events | Decay-weighted to reduce monopolization of old viral hits |
| Experience safety rating | `static` | at moderation review | Safety moderation pipeline | Values: ALL_AGES, 9+, 13+, 17+; used as hard filter |
| Player age group | `static` | at account creation | User profile | Derived from birth year; COPPA-regulated for < 13 |
| Creator reputation score | `batch` | daily | Aggregated community reports + moderation actions | Low-reputation creators deprioritized in recommendations |

#### Label Definition
- **Label:** Session start event (player clicks to join a game) as positive; impression without join as negative
- **Secondary label:** Session duration > 5 min (quality signal; guards against clickbait titles)
- **Collection strategy:** Implicit feedback via session logs; join event is immediate; duration observed after session ends
- **Positive/negative ratio:** ~1:30 (most impressions don't result in joins)
- **Label delay:** Join is immediate; session quality (duration) observed within 30–60 min
- **Bias risks:**
  - **Monopolization bias:** Top-10 games generate most training signal → model learns to recommend them → feedback loop → creator ecosystem collapses for new creators. Mitigate: creator_equity regularization in loss + exploration budget
  - **Age group bias:** Most training signal comes from 13–25 age group; model may underperform for under-13 players (COPPA restrictions limit data collection). Evaluate per-age-group recall separately
  - **Content type bias:** "Obby" (obstacle course) games dominate engagement for young players; model may over-index on a single genre

#### Storage Engine Choice

| Layer | Engine | Justification |
|---|---|---|
| Feature store (online, player) | Redis Cluster | Sub-ms reads; TTL for session signals; sharded by player_id |
| Friend presence (real-time) | Redis pub/sub (presence service → feature store) | < 30s latency; ephemeral; players go online/offline frequently |
| Feature store (offline) | Delta Lake on S3 | ACID; time-travel for training reproducibility |
| Experience embeddings | Redis (key: experience_id → embedding) | ~40M experiences × 1KB = 40GB; cache top-1M active; rest on S3 |
| Safety ratings | Redis (key: experience_id → {safety_rating, age_floor}) | Small dataset; fast lookup at filter stage |
| Training data | Parquet on S3 (date-partitioned) | Columnar; Spark-efficient |
| Logs / labels | Kafka → Flink → Delta Lake | Streaming; Flink joins session starts with impression events |
| Model artifacts | S3 + internal model registry | Versioned; per-age-group variant tracking |

#### Online vs. Offline Split

```
Offline (batch, Spark)                              Online (real-time, < 150ms)
────────────────────────────────────────────        ──────────────────────────────────────────────
Session logs → Spark → player embeddings            Request: player_id + device + surface
Experience descriptions → NLP + CLIP → embeddings  Redis: player embedding + session history (3ms)
Spark: experience popularity (decay-weighted)       Redis: friend activity (30s freshness) (2ms)
Spark: creator reputation scores                    Safety filter: hard-remove age-inappropriate (1ms)
Daily training: two-tower + DCN ranking             FAISS ANN: retrieve top-200 experiences (8ms)
Fairness audit: creator equity slot distribution    DCN ranking: score 200 candidates (15ms)
Champion/challenger: recall + diversity + equity    Creator equity injection: ensure diversity (2ms)
                                                    Response + async Kafka log
```

**Safety filter is a hard filter (not a soft feature):**
- Before any ML ranking, filter out all experiences with `age_floor > player_age_group`
- This filter runs *before* candidate retrieval — the FAISS index is age-partitioned so under-13 players never retrieve 13+ experiences
- Roblox legal/COPPA requirement: this cannot be a model probability; it must be deterministic

#### Schema

```
Player: {
  player_id:          string
  age_group:          enum[UNDER_13, TEEN, ADULT]   # derived from birth year; COPPA-regulated
  history_embedding:  float[256]
  session_recent:     string[5]                      # experience_ids; TTL 30 min
  updated_at:         timestamp
}

Experience: {
  experience_id:      string
  creator_id:         string
  content_embedding:  float[256]
  genre_tags:         string[]
  safety_rating:      enum[ALL_AGES, AGE_9_PLUS, AGE_13_PLUS, AGE_17_PLUS]
  age_floor:          int                            # minimum player age (hard filter)
  concurrent_players: int                            # real-time via presence service
  creator_reputation: float                          # [0, 1]; updated daily
  updated_at:         timestamp
}

SessionEvent: {
  player_id:          string
  experience_id:      string
  event_type:         enum[IMPRESSION, JOIN, LEAVE]
  session_duration_s: int?                           # only for LEAVE
  rank_position:      int
  timestamp:          timestamp
}
```

> **Roblox rubric:** Safety rating is a separate field with a hard age_floor (not a soft score). Creator reputation is a first-class feature. Friend activity freshness (30s) is a first-class NFR.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** Session events → Kafka → Flink → Delta Lake; join impression + join + leave events to construct training samples
- **Feature engineering:**
  - Player embedding: mean-pool of content embeddings for last 200 joined experiences (Spark, hourly)
  - Experience content embedding: CLIP on screenshots + SentenceTransformer on description → 256-d (daily)
  - Popularity with decay: `pop_score = Σ joins_i × e^(-λ × days_since)`, λ = 0.05 — prevents old viral games from permanently dominating
  - Creator equity score: `creator_equity = (creator's recommendation slots) / (creator's eligible impressions)` — baseline: each creator should get proportional representation
- **Train/val/test split:** Time-based — train D-30 to D-1; val D-1; test D-0 held-out players (not held-out time — tests generalization to new players, which is the key cold-start challenge)
- **Orchestration:** Spark DAG; idempotent; daily full retrain + hourly online learning on recent joins

#### Model Architecture

**Stage 1 — Candidate Retrieval (Two-Tower):**
- Identical pattern to Netflix recommendation (intentionally — demonstrates pattern recognition across companies)
- Player tower: `[history_embedding, session_recent_mean, age_group_embedding, friend_overlap_embedding]` → 128-d
- Experience tower: `[content_embedding, genre_tag_embedding, safety_rating_embedding]` → 128-d (safety_rating as embedding, not filter — the hard filter runs at retrieval index level)
- FAISS index: separate indexes per age_group (under-13 index contains ONLY all-ages + 9+ experiences; hard partitioning)
- Retrieval: top-200 per player request

**Stage 2 — Ranking (DCN-v2):**
- Features: player × experience cross features (history_embedding · content_embedding, genre affinity, friend_plays_this_experience)
- Multi-task: P(join) + P(session_duration > 5 min) — quality guardrail against clickbait
- Loss: `L = 0.7 × BCE(join) + 0.3 × BCE(quality)` — quality weight chosen to reduce clickbait without hurting recall

**Creator equity injection (post-ranking):**
- After DCN ranking produces sorted list, apply slot-level creator equity injection
- If creator_equity_score for creator C < 0.5 of global baseline → boost one experience from C into top-20
- Maximum: 5% of slots reserved for diversity injection (ε-greedy exploration for new experiences + underrepresented creators)
- Trade-off: measurable impact on primary metric (join rate) is −0.3% — acceptable for creator ecosystem health

#### Training Infrastructure
- **Framework:** PyTorch + DDP (two-tower); LightGBM (ranking as CPU-efficient alternative to DCN-v2 for initial launch)
- **Scale:** Two-tower: 16× A10G, ~3hr/run; DCN-v2 ranking: 8× A10G, ~2hr/run
- **Mixed precision:** bfloat16
- **Eval metrics:** Recall@200 (retrieval), NDCG@10 (ranking), join rate lift vs. popularity baseline, creator equity Gini coefficient (goal: < 0.6)

---

### 3b. Online Serving

#### Inference Path

```
Player App → API Gateway
  → Recommendation Service
      ├─ Safety Pre-filter
      │    └─ Route to age-appropriate FAISS index (under-13 / teen / adult)
      ├─ Feature Fetch
      │    ├─ Redis: player embedding + session history (3ms)
      │    └─ Redis: friend presence (games friends playing now) (2ms)
      ├─ Candidate Retrieval
      │    └─ FAISS ANN on age-appropriate index → top-200 (8ms)
      ├─ Ranking
      │    └─ DCN-v2: score 200 candidates (15ms, GPU)
      ├─ Post-ranking
      │    ├─ Friend boost: elevate games friends are playing (+3 positions)
      │    ├─ Creator equity injection: slot diversity for underrepresented creators
      │    └─ Business rules: remove experiences in maintenance, full servers
      └─ Response: top-N ranked experiences + async Kafka log
```

#### Latency Budget Breakdown

| Step | p50 | p99 | Notes |
|---|---|---|---|
| Safety pre-filter + index routing | < 1ms | 1ms | Deterministic; Redis lookup |
| Feature fetch (Redis) | 4ms | 10ms | Parallel reads |
| FAISS ANN retrieval (age-partitioned) | 6ms | 15ms | Per-age index; smaller index = faster search |
| DCN-v2 ranking (200 candidates) | 10ms | 30ms | GPU; TensorRT |
| Post-ranking (friend boost + equity) | 2ms | 5ms | CPU arithmetic |
| Network + serialization | 5ms | 15ms | gRPC |
| **Total** | **28ms** | **76ms** | Budget: ≤ 150ms p99 ✓ |

#### Caching Strategy
- **Pre-computed recommendation cache:** (player_id + hour_bucket) → ranked list; TTL 5 min
- **Friend presence cache (Redis pub/sub):** Updated every 30s via presence service; model server subscribes to updates
- **FAISS index (per age group):** Rebuilt daily; blue/green swap; old index kept warm 2hr post-swap

---

### 3c. Monitoring

#### Drift Detection

| Signal | Method | Threshold | Action |
|---|---|---|---|
| Join rate vs. control (5% holdout) | Rolling 24hr | > −3% relative | Page on-call; compare to holdout |
| Creator equity Gini coefficient | Daily | > 0.65 (monopolization signal) | Increase equity injection rate; alert creator team |
| Age filter breach rate | Daily audit | Any under-13 user sees 13+ experience (should be 0) | Immediate rollback; safety incident |
| Cold-start recall (new player NDCG) | Daily on new accounts | NDCG@10 < 0.2 for accounts < 7 days old | Content embedding quality degradation; retrain |
| Friend presence feature lag | Real-time | > 60s staleness on friend activity | Presence service alert; degrade to batch friend overlap feature |
| Experience embedding drift | PSI weekly | PSI > 0.25 on experience embedding centroids | Retrain experience embedding pipeline |

#### Shadow Scoring
- Challenger model on 2% of players; join rate + session quality logged; promoted after 7 days with +1% NDCG@10 + no creator equity regression + no safety breach

#### A/B Holdout Design
- **Unit:** `player_id` (consistent cross-device)
- **Holdout:** 5% permanent holdout (popularity-based ranking; tests value of personalization)
- **Stratification:** Stratify by age_group and player_tenure (new vs. returning players)
- **Primary metric:** Session time per recommendation session (guards against join-then-quit clickbait)
- **Guardrail metrics:** Creator equity Gini, safety filter breach rate (must be 0), session quality rate

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Redis (player features) unavailable | No personalization | Retry ×1; circuit breaker | Popularity-based ranking (no personalization; still age-filtered) |
| FAISS age-group index unavailable | No candidate retrieval | Reload from S3 (< 2s); blue/green standby | Pre-cached recommendation list from S3 (hourly batch; age-appropriate) |
| Friend presence service lag | Friend activity signals stale | Monitor staleness; alert at > 60s | Use batch friend overlap features (less real-time but still useful) |
| Age filter bug (safety breach) | Under-13 players see age-inappropriate content | Safety incident P0; automatic rollback of any model change deployed in last 24hr; manual review before re-deployment | Roll back to last known-safe model; age filter tested in CI pipeline with explicit test cases |
| DCN-v2 ranking failure | No ML ranking | Pre-ranked lists from S3 (hourly, personalized by play history) | Popularity ranking within age group (rule-based; no ML) |
| Creator equity injection disabled | Monopolization by top games | Equity injection runs as separate post-processing service; failure is isolated; alert creator ops team | Top-N ranked list served without equity injection; monitor Gini coefficient |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > 1% 5xx → popularity ranking fallback; alert on-call
- **Latency:** Trip at p99 > 200ms sustained 3min → disable DCN-v2; serve retrieval output by similarity score
- **Safety breach:** Any under-13 exposure to 13+ content → immediate full rollback; P0 incident

#### Degraded-Mode Behavior
1. **Level 1** — Full pipeline: two-tower retrieval + DCN ranking + friend boost + creator equity — optimal
2. **Level 2** — Two-tower retrieval only (no DCN ranking; sort by embedding similarity + popularity)
3. **Level 3** — Age-filtered popularity ranking (rule-based; no ML; no personalization)

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: 80M
> - Recommendation requests per DAU: 10/day (home page loads, Discover refreshes)
> - Peak QPS multiplier: 2× (after-school peak: 3–6 PM EST)
> - Experience embeddings: 256 floats = 1KB; top-1M active experiences in Redis = 1GB
> - FAISS index: 3 age-group indexes × 1M active experiences × 256-d = ~3GB total (trivially in memory)
> - Log retention: 90 days hot, 1 year cold

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | 80M × 10 / 86,400 | **~9,260 QPS** |
| Peak QPS | 9,260 × 2× | **~18,500 QPS** |
| Redis reads/s | 18,500 × 3 reads | **~55K reads/s** |
| Experience embedding storage (active) | 1M active × 1KB | **~1GB in Redis** |
| Session event ingest (Kafka) | 18,500 × 10 events (impressions + join + leave) | **~185K events/s → ~18GB/hr** |
| Training data (30 days) | 80M DAU × 10 events/day × 30 days × 200B | **~480GB** (compressed ~100GB) |
| FAISS indexes (3 age groups, top-1M) | 3 × 1M × 256-d × 4B | **~3GB total** |
| DCN-v2 serving replicas | 18,500 QPS / 500 QPS per A10G | **~37 A10G GPUs** |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] **Cross-platform creator content:** Many Roblox experiences have associated YouTube walkthroughs and TikTok trends. Should external popularity signals inform recommendation? External content may be age-inappropriate → legal/safety review required
- [ ] **Multiplayer session recommendation:** "Join your friend's game" is a strong conversion trigger. Should the recommendation system have a dedicated "social mode" that prioritizes friend-joinable active sessions over personalized single-player recommendations?
- [ ] **Creator tier progression:** New creators are bootstrapped with random exposure; could a UCB1 bandit approach better balance exploration for new creators vs. exploitation for proven hits?
- [ ] **Under-13 data collection:** COPPA restricts behavioral data collection for under-13 users. Can we use parental consent flow for richer data? Legal review required for any changes to under-13 data handling

#### Company Rubric Gaps (self-assessment)
- [x] Requirements: age-appropriate filtering as hard requirement; creator equity as guardrail metric; friend presence freshness (30s)
- [x] Data Modeling: age-partitioned FAISS indexes; creator reputation in schema; safety rating as hard filter field
- [x] ML Pipeline: creator equity injection post-ranking; two-objective loss (join + session quality)
- [x] Failure Modes: safety breach = P0 rollback (not just an alert); creator equity failure isolated to separate service
- [x] Capacity: per-age-group FAISS index size calculated; after-school peak explicitly modeled

#### Recommended Follow-up Problems
- Netflix Homepage Recommendation — identical two-tower pattern; compare cold-start strategies
- Trust & Safety: Content Moderation for UGC (Roblox's 40M user-created games require moderation at scale)

---

## 7. References

| Resource | Type | Relevance |
|---|---|---|
| Roblox Tech Blog: "Recommendation System at Roblox" | Blog | Production recommendation architecture; cold-start strategy; creator diversity |
| Covington et al., "Deep Neural Networks for YouTube Recommendations" (RecSys 2016) | Paper | Foundational two-tower recommendation; directly analogous to Roblox's problem |
| Burke, "Hybrid Recommender Systems: Survey and Experiments" (2002) | Paper | Hybrid content + collaborative filtering; applicable to cold-start handling |
| Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021) | Paper | CLIP for game screenshot embeddings; enables cold-start for new experiences |
| Wang et al., "DCN V2: Improved Deep & Cross Network" (WWW 2021) | Paper | DCN-v2 for ranking stage |
| Salimans et al., "Improved Techniques for Training GANs" (2016) | Paper | Evaluation metrics for generative diversity; Gini coefficient adapted for creator equity |
| Children's Online Privacy Protection Act (COPPA) | Regulation | Data collection restrictions for under-13 users; hard filter requirements |
| Zhao et al., "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019) | Paper | Multi-task ranking; session quality as secondary objective |
| Mehrotra et al., "Towards a Fair Marketplace: Counterfactual Evaluation of the trade-off between Relevance, Fairness & Satisfaction in Recommendation Systems" (CIKM 2018) | Paper | Creator equity / fairness in recommendation; Gini-based fairness metrics |
| Lattimore & Szepesvári, "Bandit Algorithms" (Cambridge, 2020) | Book | UCB1 and ε-greedy exploration; applicable to new creator bootstrapping |
