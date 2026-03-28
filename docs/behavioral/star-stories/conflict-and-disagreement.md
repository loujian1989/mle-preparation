# Conflict & Disagreement

P1: Meta (✓✓), Netflix (✓✓). Asked at every company.

---

## Common Question Phrasings

- "Tell me about a time you disagreed with your manager or a senior stakeholder."
- "Describe a situation where you had a strong technical opinion that wasn't initially accepted."
- "Tell me about a time you pushed back on a product or engineering direction."
- "Describe a conflict with a peer that you had to resolve."
- "Tell me about a time you changed someone's mind."

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "I disagreed with my manager. I explained my view, we talked it out, they agreed." | "The org was moving toward a technically risky direction. I built a data-backed case, got buy-in from 2 other teams, and changed the roadmap — including walking my skip-level through the trade-offs." |
| "I raised the concern in our team meeting." | "I identified the right forum (not just my manager), prepared the counter-proposal with specifics, and gave the stakeholder a face-saving path to change course." |
| Story resolves in 1 conversation | Story spans multiple weeks, multiple stakeholders, multiple escalations |
| "I was right and they were wrong" | "I was right on the technical point but wrong on the rollout approach — I updated my position mid-conflict when I got new information." |

---

## Story Template 1: Wrong Technical Bet at Org Level

**Situation:** Your organization is committing to a technical approach — a new model architecture, a platform migration, an infrastructure decision — that you believe has a critical flaw (scalability cliff, unvalidated assumption, hidden dependency). The decision is already in motion.

**Task:** You are the person with the most context to identify the risk. Nobody is explicitly asking you to challenge it. Taking this on means disagreeing with leaders who have already committed.

**Action:**
1. Quantify the risk: write a short internal doc with specific failure scenarios + cost estimates (not just "it might not work")
2. Find allies: identify 1-2 engineers or PMs who share the concern and brief them before any group meeting
3. Choose the right forum: skip the team standup; request a focused technical review with the decision-maker and relevant cross-functional stakeholders
4. Present a concrete alternative: never bring only a problem — bring "here's the alternative, here's the cost, here's how we migrate"
5. Give a face-saving path: frame as "new information" not "you were wrong" — "we now have data we didn't have before"

**Result:** [Fill in: did the decision change? partially? what was the second-order outcome — relationship, trust, precedent set?] What you'd do differently: X.

**Best fit for:**
- "Tell me about disagreeing with a stakeholder"
- "Tell me about changing someone's mind"
- "Tell me about a time you showed leadership outside your role"

**Company fit:**
- **Meta**: Lead with "move fast" tension — the org wanted to ship fast, you slowed down to avoid a later rewrite
- **Netflix**: Lead with "informed captain" — you had the context, you made the call, you owned the outcome
- **OpenAI**: Lead with intellectual honesty — you publicly updated your view when new evidence came in

**Watch out for:** The story falls flat if you "raised a concern in Slack and moved on." You must have actively driven the resolution, even if you didn't get your way.

---

## Story Template 2: Disagreement with a Peer on Model/System Design

**Situation:** You and a peer (another Staff or senior engineer) have fundamentally different approaches to a shared system — one of you wants to build a complex custom solution; the other wants to use an existing imperfect tool. The disagreement is blocking progress.

**Task:** Unblock the team without bulldozing your peer and without compromising on the right technical outcome.

**Action:**
1. Separate the technical question from the interpersonal one: write up both approaches in a shared doc with explicit trade-offs, no advocacy
2. Define decision criteria first: agree on what success looks like before arguing about the solution
3. Run a time-boxed spike: if the disagreement is empirical ("ours will be faster"), test it instead of debating it
4. Make the decision explicit: produce a written ADR (Architecture Decision Record) — no verbal agreements that drift later

**Result:** [Fill in: which approach was chosen, or was it a synthesis? what did the team learn? did it set a precedent for future decisions?]

**Best fit for:**
- "Tell me about a technical conflict with a peer"
- "Tell me about a time you had to compromise"

**Company fit:**
- **Shopify**: Lead with "decision log" framing — the value was that you documented the reasoning, not just the outcome
- **Stripe**: Lead with precision — you ran the empirical test instead of debating subjectively

**Watch out for:** Don't describe it as "they came around to my view." Show the synthesis, or show you were wrong about part of it.

---

## Story Template 3: Escalating a Decision You Disagree With

**Situation:** You disagree with a decision, you've made your case, and the decision-maker has heard you but is proceeding anyway. You believe the decision is genuinely wrong.

**Task:** Decide when to escalate vs. commit-and-execute. This is the hardest version of the conflict question.

**Action:**
1. Define your escalation criteria: "I'll escalate if this could cause irreversible harm (data loss, user trust breach, regulatory risk). I won't escalate over a directional disagreement where reasonable people differ."
2. If escalating: go to your skip-level with facts, not feelings. "Here are 3 scenarios where this fails, with cost estimates. I'm not asking you to override — I'm asking for a second technical review."
3. If not escalating: explicitly commit. "I disagree but I'll commit. Here's the rollback plan I'm building so we can recover quickly if my concern is right." Then document your concern in writing.
4. After resolution: follow up with the outcome. If you were wrong, say so explicitly. If you were right, don't gloat — the relationship is the asset.

**Result:** [Fill in: what happened? did the decision go well or badly? what was preserved in either case?]

**Best fit for:**
- "Tell me about a time you disagreed but had to comply with a decision"
- "Tell me about escalating a concern"
- "Tell me about a time you were wrong"

**Company fit:**
- **Netflix**: This maps directly to "freedom and responsibility" — you had the freedom to disagree, you took the responsibility of documenting the risk and building the fallback
- **Meta**: Maps to "be open" — you disagreed loudly but committed once the decision was final

**Watch out for:** If your escalation story is "I went over my manager's head because they were wrong," that's a red flag. The story needs to show judgment about when escalation is warranted.
