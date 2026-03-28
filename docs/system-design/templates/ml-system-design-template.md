# <Problem Name> — ML System Design

**Domain:** `<ranking | search | real-time-ml | marketplace | ads | other>`
**Target Company:** `<Meta | Netflix | Uber | Shopify | Other>`
**Difficulty Bar:** `<L5 | L6 | L7>`
**Date:** <YYYY-MM-DD>
**Related Designs:** `<link to related doc if any>`

---

## 0. Scorecard

> Fill this in last. Score each axis 1–5 after completing all sections.

| Axis | Score | Weakest Point |
|---|---|---|
| Requirements | ★☆☆☆☆ | <what's missing> |
| Data Modeling | ★☆☆☆☆ | <what's missing> |
| ML Pipeline | ★☆☆☆☆ | <what's missing> |
| Failure Modes | ★☆☆☆☆ | <what's missing> |
| Capacity | ★☆☆☆☆ | <what's missing> |

**Overall:** `STRONG HIRE | HIRE | NO HIRE`
**Top Gap:** <single most important axis to improve>

---

## 1. Requirements

#### Functional Requirements
1. <primary user-facing capability>
2. <secondary capability>
3. <tertiary capability>

#### Non-Functional Requirements
| Dimension | Target | Notes |
|---|---|---|
| Latency (p99 serving) | <Xms> | <assumption or constraint> |
| Availability | <99.X%> | <SLA source> |
| Consistency | <eventual / strong / read-your-writes> | <trade-off accepted> |
| Throughput | <X QPS peak> | <traffic pattern> |
| Freshness | <feature staleness budget> | <real-time vs. batch> |

#### Scale Numbers (state upfront)
- **DAU / MAU:** <X>
- **Peak QPS:** <X>
- **Data volume (raw):** <X TB/day>
- **Model inference calls/day:** <X>

#### Out of Scope
- <explicit exclusion 1>
- <explicit exclusion 2>

> **Company watch-fors:**
> - **Meta:** State scale mechanisms explicitly (sharding strategy, geographic distribution). "It's distributed" is not enough — name the consistency/throughput trade-off.
> - **Netflix:** Tie every non-functional requirement to a user or business metric. Latency SLA should reference what user experience breaks if violated.
> - **Uber:** Real-time freshness of driver supply/demand signals is a first-class requirement. State sub-second where applicable.
> - **Shopify:** Present ≥2 design options for key NFRs with explicit reasoning before committing. Document what was rejected and why.

---

## 2. Data Modeling

#### Input Signals

| Feature | Type | Freshness | Source | Notes |
|---|---|---|---|---|
| <feature name> | `<real-time / batch / static>` | <staleness budget> | <system name> | <why it matters> |
| <feature name> | `<real-time / batch / static>` | <staleness budget> | <system name> | <why it matters> |

#### Label Definition
- **Label:** <what you're predicting>
- **Collection strategy:** <how labels are gathered — implicit feedback, explicit ratings, delayed conversion, etc.>
- **Positive/negative ratio:** <X:1>
- **Label delay:** <how long before label is observed>
- **Bias risks:** <position bias, selection bias, survivorship bias — as applicable>

#### Storage Engine Choice
| Layer | Engine | Justification |
|---|---|---|
| Feature store (online) | <Redis / DynamoDB / Cassandra> | <low-latency reads, TTL, schema flexibility> |
| Feature store (offline) | <Hive / BigQuery / Delta Lake> | <analytical workloads, cheap storage> |
| Model artifacts | <S3 / GCS> | <versioned, cheap> |
| Training data | <Parquet on S3 / BigQuery> | <columnar, partitioned by date> |
| Logs / labels | <Kafka → Hive / BigQuery> | <streaming ingest, batch join> |

#### Online vs. Offline Split

```
Offline (batch)                        Online (real-time)
───────────────────────────────        ──────────────────────────────
Raw events → feature pipeline          Request → feature fetch (<10ms)
Parquet/Delta store                    Feature store (Redis/Cassandra)
Training jobs (daily/weekly)           Model server (gRPC, <Xms p99)
Evaluation + champion/challenger       Response → logging → label join
```

#### Schema (key entities only)
```
<Entity>: {
  id:         <type>
  <field>:    <type>   # <purpose>
  <field>:    <type>
  created_at: timestamp
}
```

> **Company watch-fors:**
> - **Meta:** Address hot partition risk on high-cardinality keys (user_id, item_id). Name the sharding key and replication factor.
> - **Netflix:** Explicitly distinguish online/offline feature consistency — stale offline features during serving are a failure mode Netflix probes.
> - **Uber:** Use H3 hexagonal indexing for any geospatial features. City-level heterogeneity should be reflected in feature schema.
> - **Shopify:** SOLID — no God objects. Schema should have clear single responsibilities per entity.

---

## 3. ML Pipeline

### 3a. Offline Training

#### Data Pipeline
- **Ingestion:** <raw source → transformation → storage>
- **Feature engineering:** <key transforms: embeddings, aggregations, normalization>
- **Train/val/test split strategy:** <time-based split recommended — state the cutoff logic>
- **Pipeline orchestration:** <Airflow DAG / Metaflow / Spark job — idempotent by default>

#### Model Architecture
| Option | Pros | Cons | Decision |
|---|---|---|---|
| <Model A> | <+> | <-> | <chosen / rejected> |
| <Model B> | <+> | <-> | <chosen / rejected> |

**Selected architecture:** <name>
**Justification:** <why>

#### Training Infrastructure
- **Framework:** <PyTorch + DDP/FSDP | TensorFlow | XGBoost>
- **Scale:** <X GPU nodes, X hours/run>
- **Mixed precision:** <yes/no — reason>
- **Gradient checkpointing:** <yes/no — reason>
- **Eval metric:** <primary metric + why it proxies business goal>

---

### 3b. Online Serving

#### Inference Path
```
Client → Load Balancer → Feature Fetch (Redis, <5ms)
                       → Model Server (gRPC)
                       → Post-processing / re-ranking
                       → Response + async logging
```

#### Latency Budget Breakdown
| Step | Budget | Actual (target) |
|---|---|---|
| Feature fetch | <Xms> | <Xms p99> |
| Model inference | <Xms> | <Xms p99> |
| Post-processing | <Xms> | <Xms p99> |
| Network + overhead | <Xms> | <Xms p99> |
| **Total** | **<Xms** | **<Xms p99>** |

#### Caching Strategy
- **What is cached:** <candidate set / embeddings / ranked results>
- **Cache key:** <user_id + context hash / item_id>
- **TTL:** <X seconds — justify based on feature freshness>
- **Cache hit rate target:** <X%>

---

### 3c. Monitoring

#### Drift Detection
| Signal | Method | Alert Threshold | Action |
|---|---|---|---|
| Input feature drift | PSI / KS test | PSI > 0.2 | Retrain trigger |
| Label distribution shift | Chi-squared | p < 0.01 | Investigate + retrain |
| Prediction score drift | Rolling mean ± 2σ | > 10% shift | Shadow model comparison |
| Business metric | Experiment holdout | < -X% vs. control | Rollback |

#### Shadow Scoring
- **Shadow model:** <challenger model runs in parallel, no traffic impact>
- **Comparison cadence:** <daily offline comparison on held-out slice>
- **Promotion criteria:** <primary metric improvement + no regression on guardrail metrics>

#### A/B Holdout Design
- **Unit of randomization:** <user_id | session_id | request_id>
- **Holdout size:** <X% — justify>
- **Primary metric:** <metric name + minimum detectable effect>
- **Guardrail metrics:** <latency p99, error rate, revenue per user>
- **Duration:** <X weeks — power calculation basis>

> **Company watch-fors:**
> - **Meta:** Flink/Spark feature pipeline; p99 online serving <100ms. State mixed precision and gradient checkpointing decisions explicitly.
> - **Netflix:** Monitoring is not an afterthought — answer "how do you know it's working in prod?" before they ask. Drift detection + shadow scoring must be in the design from the start.
> - **Uber:** Prediction intervals required for any regression output (point estimates alone are insufficient). State distributional shift handling.
> - **Shopify:** Testability — monitoring hooks should be injectable, not hardcoded. Written for unit testability by default.

---

## 4. Failure Modes

| Scenario | Impact | Mitigation | Fallback |
|---|---|---|---|
| Feature store unavailable | Stale / missing features | Cache last-known-good values with TTL | Default features (global averages) |
| Model server timeout | No predictions | Retry with backoff (2 attempts, 50ms timeout) | Rule-based fallback (popularity ranking) |
| Training job failure | Stale model | Alert + auto-retry; champion model stays live | Keep current champion; alert on-call |
| Label pipeline delay | Training data gap | Detect missing partitions; block retraining until resolved | Log gap; backfill when recovered |
| <scenario> | <impact> | <mitigation> | <fallback> |

#### Circuit Breaker Thresholds
- **Error rate:** Trip at > <X>% 5xx over 60s rolling window
- **Latency:** Trip at p99 > <Xms> sustained for > <Y> seconds
- **Recovery:** Half-open after <Z> seconds; full-open on first success

#### Degraded-Mode Behavior
When ML is unavailable:
1. <Fallback level 1 — e.g., cached ranked list from last successful run>
2. <Fallback level 2 — e.g., popularity-based ranking, no personalization>
3. <Fallback level 3 — e.g., static default list>

> **Company watch-fors:**
> - **Meta:** Geographic distribution failure — state what happens when a regional data center is degraded.
> - **Netflix:** Define the full fallback chain explicitly. Netflix will ask for it.
> - **Uber:** Cascading marketplace effects — a failed ETA model affects driver dispatch, pricing, and matching. State which systems depend on this model's output.
> - **Shopify:** Fallbacks should be explicitly tested (injected, not mocked at the DB level).

---

## 5. Capacity Estimates

> **Assumptions:**
> - DAU: <X>
> - Peak QPS: <X> (assume <Y>× of average; peak at <time of day / event>)
> - Average request payload: <X KB>
> - Feature vector size: <X floats>
> - Model size: <X MB>
> - Log retention: <X days>

#### Estimates

| Metric | Calculation | Result |
|---|---|---|
| Avg QPS | DAU × requests/day / 86,400 | <X> QPS |
| Peak QPS | avg × <Y>× multiplier | <X> QPS |
| Feature store reads/s | peak QPS × features/request | <X> reads/s |
| Feature storage (online) | users × features × <X bytes> | <X GB> |
| Training data (daily) | events/day × <X bytes>/event | <X GB/day> |
| Training data (1 year) | daily × 365 | <X TB> |
| Model artifact storage | <X versions> × <X MB>/model | <X GB> |
| Serving latency (p50) | <estimate based on model size + hardware> | <Xms> |
| Serving latency (p99) | p50 × <Y>× tail factor | <Xms> |
| Training compute | <X examples> / <X examples/GPU/s> | <X GPU-hours/run> |
| Serving replicas needed | peak QPS / <X QPS/replica> | <X replicas> |

---

## 6. Open Questions & Follow-ups

#### Unresolved Design Decisions
- [ ] <question 1 — e.g., should we use a two-tower model or DLRM?>
- [ ] <question 2>
- [ ] <question 3>

#### Company Rubric Gaps (self-assessment)
- [ ] Requirements: scale mechanisms named explicitly?
- [ ] Data Modeling: online/offline feature consistency addressed?
- [ ] ML Pipeline: monitoring designed upfront, not as afterthought?
- [ ] Failure Modes: full fallback chain defined?
- [ ] Capacity: all estimates have stated assumptions?

#### Recommended Follow-up Problems
- <related problem to practice next>
- <related problem to practice next>
