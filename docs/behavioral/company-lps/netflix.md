# Netflix — Behavioral Prep

Behavioral round: dedicated culture-fit interview, usually 45–60 min. Explicitly grounded in the Netflix Culture Deck.

---

## Netflix's Core Culture Tenets (Relevant to MLE)

- **Judgment**: "You make wise decisions despite ambiguity." You don't wait for certainty or permission.
- **Communication**: "You are concise and articulate. You listen as much as you speak."
- **Impact**: "You accomplish amazing amounts of important work." Volume + quality both expected.
- **Curiosity**: "You learn rapidly and eagerly. You seek to understand our strategy, market, and subscribers."
- **Innovation**: "You re-conceptualize issues to discover practical solutions."
- **Courage**: "You say what you think, even when it's uncomfortable. You are willing to be critical of the status quo."
- **Honesty**: "You are known for candor and directness. You are non-political."
- **Selflessness**: "You seek what is best for Netflix, not yourself or your team."

**The Keeper Test**: "If X wanted to leave, would I fight to keep them?" — Netflix actively manages out people who are merely adequate. Your stories must demonstrate you're irreplaceable, not just competent.

---

## What They're Actually Testing

- **Freedom & responsibility**: "We give you enormous freedom. We expect enormous responsibility." Stories that show you operated without supervision AND delivered excellent outcomes are the target.
- **High performance bar**: Netflix doesn't want hard workers — they want efficient, high-judgment performers. "We want brilliant jerks to be replaced by brilliant teammates."
- **Candor**: Do you give and receive direct feedback? Do you say uncomfortable things in the room? Netflix is skeptical of politically savvy answers that avoid the point.
- **Context not control**: Did you operate from first principles rather than waiting for direction? Did you build context before taking action?
- **Informed captain**: When a decision had to be made and you were the most informed person in the room, did you make the call?

---

## Question Patterns

| Question | Tenet it maps to | Story to tell |
|---|---|---|
| "Tell me about a time you gave someone hard feedback directly." | Courage + Honesty | `technical-leadership.md` — Template 3 (growing the engineer; the specific gap diagnosis is the "hard feedback") |
| "Tell me about a situation where you acted like an informed captain." | Judgment | `ambiguity-and-scope.md` — Template 1 (zero-to-one, you made the calls) |
| "Tell me about a time you were told you were wrong and updated your view." | Curiosity + Honesty | `failure-and-learning.md` — Template 2 |
| "Tell me about a time you disagreed with a decision and how you handled it." | Courage + Communication | `conflict-and-disagreement.md` — Template 3 (escalation + commit) |
| "Tell me about a time you delivered something with no one telling you what to do." | Freedom & Responsibility | `ambiguity-and-scope.md` — any template |
| "Tell me about raising the bar on quality." | Impact | `technical-leadership.md` — Template 1 (tech debt) or Template 2 |
| "Tell me about a high-stakes situation and how you handled the stress." | Judgment + Impact | `failure-and-learning.md` — Template 1 (production incident) |

---

## The Keeper Test Framing

Netflix interviews sometimes ask directly: "If you left tomorrow, what would be irreplaceable about you?" or probe for it implicitly. Be ready to articulate:

- **Technical uniqueness**: What do you know or can you do that would take 6+ months for someone else to rebuild?
- **Organizational knowledge**: What cross-team relationships or institutional knowledge do you carry?
- **Judgment in your domain**: What decisions do others defer to you on that are beyond pure execution?

Bad answer: "I work hard and care about quality." Everyone says this.
Staff answer: "I'm the person who can bridge ML research and production reliability — I understand both what's mathematically valid and what will survive at 1M QPS. That's rare and took 3 years to build."

---

## Signals That Differentiate Staff from Senior

- **Proactiveness**: Senior waits to be assigned problems. Staff identifies problems that aren't on anyone's radar yet.
- **Candor level**: Netflix specifically tests whether you'll say things that are uncomfortable. Be willing to name the hard part of your story directly.
- **No hedging**: "It went reasonably well" is a Netflix red flag. Be specific and direct about outcomes — including bad ones.
- **Responsibility for your whole domain**: Did you treat your system/team/area as if you owned it fully — not just your assigned tasks?

---

## Red Flags (Lower the Bar)

- "I checked with my manager before making that call" — autonomy is expected, not optional
- Generic positive language without specifics — "I'm very collaborative" without evidence
- Stories with no failure, no hard decision, no tension — life is not that easy and Netflix knows it
- Not knowing your numbers — Netflix expects quantified outcomes
- Political language — "I wanted to make sure everyone was aligned" without the hard conversation that created alignment

---

## Core Ads Algorithms Team-Specific Angle

The MLS5 Ads role (Core Ads Algorithms team) has a distinct behavioral layer on top of the standard Netflix culture probes. The team is building Netflix's in-house bidding algorithms — expect questions that probe **advertiser vs. member tension**, **measurement rigor**, and **building from scratch in a new domain**.

### Ads Team-Specific Question Patterns

| Question | What it's probing | Story angle |
|---|---|---|
| "Tell me about building an ML system where two customers had conflicting objectives." | Advertiser performance vs. member experience | Bidding changes that improve ROAS but increase ad load → member churn risk. Show you held the member guardrail. |
| "Tell me about setting up an evaluation framework for a system where ground truth was delayed." | Attribution lag / measurement rigor | Any story involving delayed labels (conversion, churn, LTV). Show you built a truncated-label correction, not just waited for data. |
| "Tell me about a time you influenced the product definition, not just the ML solution." | Product partnership (role requirement: "partner with product team") | Story where you pushed back on the objective function or success metric, not just optimized the given one. |
| "Tell me about making a bold technical bet in an uncertain domain." | Courage + Innovation | Story about building something new (new auction design, new bidding algorithm) where there was no prior art at the company. |
| "Tell me about a time you caught a measurement problem before it misled stakeholders." | Measurement integrity | Attribution error, calibration drift, SRM in an A/B test — any story where you caught a data quality issue before it became a bad decision. |
| "Tell me about operating with freedom and responsibility simultaneously." | Freedom & Responsibility | Story where you had wide scope, no one checking your work, and you delivered — or caught your own mistake. |

### Ads Team Strong Framing

**On advertiser vs. member tension:**
> "The bidding algorithm I was building could improve advertiser ROAS by 15%, but our analysis showed it would increase ad load by 8% and trigger a 0.3% increase in churn. I treated churn as a hard guardrail, not a soft tradeoff. We shipped a version that improved ROAS by 9% with zero churn impact."

**On measurement integrity:**
> "We had a 30-day attribution window but were being asked to report ROAS at 7 days. I showed that 7-day ROAS was systematically 40% lower than 30-day ROAS for our category — not because the ads weren't working, but because the attribution window was wrong. I changed the measurement before stakeholders locked in on the wrong metric."

**On building from scratch:**
> "There was no prior art for this at the company. I identified the three highest-uncertainty assumptions, designed experiments to resolve each one before committing to the full architecture, and documented the decision log so the team could challenge each choice."

### Behavioral Prep Files Most Relevant to Ads Team

- `technical-leadership.md` — Template 2 (architectural decision): defining a new bidding system from scratch
- `prioritization-and-tradeoffs.md` — Template 1 (saying no with data): holding the member churn guardrail
- `cross-functional-influence.md` — Template 1 (aligning product and ML on objective): pushing back on the success metric
- `failure-and-learning.md` — Template 1 (production incident): a bidding change that caused unintended ad load increase

---

## Ads Inventory Management & Forecasting Team-Specific Angle

The MLE5 Inventory role (Ads Platform Engineering / Inventory Management & Forecasting) is **publisher-side**. It's about forecasting how much inventory Netflix has to sell, optimizing how it's allocated across buyers, and building simulations to test pricing decisions — not about bidding or campaign optimization (that's the Core Ads Algorithms team).

Expect behavioral probes around **ambiguity of building new systems**, **cross-functional alignment with sales/operations teams** (who make commitments to advertisers based on your forecasts), and **data quality and measurement rigor** (your forecast error has direct revenue consequences).

### Inventory Team-Specific Question Patterns

| Question | What it's probing | Story angle |
|---|---|---|
| "Tell me about a time your model's output directly drove a business commitment." | Ownership + accuracy under stakes | A forecast, pricing model, or allocation system where external teams made decisions (sales commitments, financial planning) based on your output. Show you owned the accuracy. |
| "Tell me about building a system from scratch with no prior art." | Boldness + ambiguity | The inventory team is new. Any story about greenfield system design with uncertain requirements maps directly. Use `ambiguity-and-scope.md` Template 1. |
| "Tell me about aligning with a non-technical stakeholder on a technical constraint." | Communication + cross-functional | Story where you had to explain to sales/operations why 100% delivery guarantee isn't possible, or why the forecast has ±15% error at 90 days. Show you converted technical nuance into actionable guidance. |
| "Tell me about a time you caught an error in your system before it caused downstream harm." | Rigor + proactiveness | Forecast calibration issue, overselling bug, or simulation validation failure caught before it led to an advertiser commitment. |
| "Tell me about making a tradeoff between delivery guarantee and revenue maximization." | Judgment | Yield optimization tradeoff: hold impressions for programmatic (higher eCPM) vs. serve to guaranteed campaign (delivery commitment). Frame the decision criteria you used. |
| "Tell me about delivering impact in a fast-moving, ambiguous environment." | Freedom & Responsibility | The job description says "new team, enormous ambitions." Any story about delivering meaningful results under uncertainty and without established playbooks. |

### Inventory Team Strong Framing

**On forecast accuracy and business impact:**
> "My forecast wasn't just a model output — it was the basis for $10M in sales commitments. I treated forecast calibration as P0 infrastructure, not an afterthought. I ran weekly calibration checks, maintained rolling error distributions by segment, and established a formal process for flagging forecasts with high uncertainty before sales locked in a commitment."

**On building simulation for inventory decisions:**
> "The team had no way to test pricing changes without running live experiments. I built a Monte Carlo simulation on historical traffic that let us estimate revenue and fill rate impact of any pricing change in hours, not weeks. Before I built it, every pricing decision was a gut call; after, it was evidence-based."

**On cross-functional alignment with sales:**
> "Sales wanted 100% delivery guarantee on every campaign. I showed them that guaranteeing 100% required holding 40% extra inventory as buffer — inventory that would go unsold. Instead I proposed 95% delivery SLAs with make-good policies, which let us sell 30% more inventory. The conversation was hard, but the data made it clear."

### Behavioral Prep Files Most Relevant to Inventory Team

- `ambiguity-and-scope.md` — Template 1 (zero-to-one): greenfield inventory system with no prior art
- `cross-functional-influence.md` — Template 1 or 3: aligning sales team on delivery SLAs vs. yield optimization
- `technical-leadership.md` — Template 2 (architectural decision): forecasting system design choices
- `prioritization-and-tradeoffs.md` — Template 1: yield vs. delivery guarantee tradeoff
