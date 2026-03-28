# Cross-Functional Influence

P1: Meta (✓✓), Pinterest (✓✓), Roblox (✓✓). The single biggest Staff bar differentiator.

---

## Common Question Phrasings

- "Tell me about a time you influenced without formal authority."
- "Describe a project where you had to get buy-in across teams."
- "Tell me about aligning stakeholders who had conflicting priorities."
- "Tell me about your biggest impact at [company/team level]."
- "Describe a situation where you had to work with product, engineering, and data science simultaneously."

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "I convinced my team to use a better library." | "I changed the inference infrastructure strategy for 3 product teams by making the case with latency data and getting 2 PMs to co-sponsor the roadmap item." |
| "I gave a tech talk." | "I identified the knowledge gap, designed the enablement program, and changed how the org makes ML deployment decisions." |
| Story = one meeting, one team | Story = multi-quarter, multi-org, lasting change |
| "I was very persuasive" | "I understood what each stakeholder needed — and gave them a different framing of the same proposal" |

---

## Story Template 1: Getting Engineering to Prioritize ML Infrastructure

**Situation:** ML team needs infrastructure work (feature store, model serving layer, training pipeline) that only a platform/infra team can build. The infra team has their own roadmap and doesn't share your team's priorities. The ML team can't ship without this dependency.

**Task:** You have no direct authority over the infra team. Getting this prioritized requires building the business case and coalition, not just asking nicely.

**Action:**
1. Quantify the cost of not having it: "Current workaround costs X engineering hours/week. At 10 teams × 3 ML engineers, that's 30 hrs/week = ~$1.5M/year in eng time."
2. Find the infra team's motivation: what are their OKRs? Frame your request as helping them hit their goals (reliability, adoption, platform usage metrics).
3. Get PM co-sponsorship: bring the product PM, not just the tech argument. PMs speak to PMs.
4. Propose a minimal version: don't ask for everything at once. "Give me 1 engineer for 6 weeks — here's the scope that unblocks us."
5. Offer reciprocal value: "We'll be the design partner, we'll write the RFC, we'll validate in production" — reduce their work, increase their credit.

**Result:** [Fill in: was it prioritized? what shipped? what was the team velocity improvement?]

**Best fit for:**
- "Tell me about influencing without authority"
- "Tell me about aligning cross-functional stakeholders"

**Company fit:**
- **Meta**: Lead with "move fast" — the workaround was slowing the org; you removed the blocker structurally
- **Uber**: Lead with "cost of the status quo" — quantify the marketplace impact of shipping late
- **Shopify**: Lead with "build for the long term" — you solved the systemic problem, not the one-off request

**Watch out for:** If you just "asked them nicely and it worked," that's a Senior story. The Staff version shows you understood their incentives and engineered alignment, not persuasion.

---

## Story Template 2: Aligning Product and ML on Metric Definition

**Situation:** Your ML model is optimizing metric A (e.g., CTR). The product team says the product is failing by metric B (e.g., retention). Data science says both are right. You realize the problem: ML, product, and data science are each measuring something different, and no one has noticed.

**Task:** Alignment on what to optimize is not a technical problem — it's an organizational problem. You're the person who can see both sides and must broker the resolution.

**Action:**
1. Surface the misalignment explicitly: write a one-page doc showing "here are the 3 things we're each measuring, here's where they diverge, here's a concrete example where optimizing A hurts B."
2. Convene the right people: don't resolve this in Slack or in 1:1s. Run a structured meeting with PM, ML lead, DS lead — end with a written agreement.
3. Propose a hierarchy of metrics: "North star = B (retention). Guardrail = A (CTR must not drop more than X%). Diagnostic = C." This is more useful than picking one.
4. Build the measurement: once agreed, own building the evaluation pipeline that tracks all three metrics in one place.
5. Document the decision: write an ADR so the next model iteration doesn't re-open this debate.

**Result:** [Fill in: how did it resolve? did it surface downstream in a launch review? what's the ongoing process?]

**Best fit for:**
- "Tell me about aligning conflicting stakeholders"
- "Tell me about your impact on how the team operates"

**Company fit:**
- **Netflix**: "Product-minded" — you proactively identified that the technical work was solving the wrong problem
- **Reddit**: "Data-driven" — you quantified the divergence; you didn't just assert there was a problem

**Watch out for:** Don't make this a "PM was wrong, I fixed it" story. The Staff version shows that the misalignment was systemic, and you built the shared framework, not just the one correct answer.

---

## Story Template 3: Driving Org-Wide Adoption of a Standard

**Situation:** You've identified a better way to do something (evaluation framework, deployment checklist, feature engineering pattern). Your team uses it. Other teams don't. The lack of standard causes repeated bugs, slower shipping, inconsistent quality.

**Task:** Standardize across the org. You have no mandate to tell other teams what to do.

**Action:**
1. Prove it works: make your version the undeniable example. If you can't show it working at your team, you have no credibility to spread it.
2. Find early adopters: identify 1 team that is already struggling with the exact problem. Offer to help them adopt it — be their guide, not their mandator.
3. Make it easy to adopt: write the runbook, create the template, reduce the activation energy. If adoption requires reading a 30-page doc, it won't happen.
4. Create visibility: present at the eng all-hands or equivalent. Get leadership endorsement. Once one VP mentions it approvingly, adoption accelerates.
5. Formalize but don't mandate: propose it as the "recommended approach," not "required approach." Optional with social proof is more durable than mandatory with resentment.

**Result:** [Fill in: how many teams adopted? what was the estimated quality/velocity impact across the org?]

**Best fit for:**
- "Tell me about raising the bar across the org"
- "Tell me about your most lasting technical contribution"

**Company fit:**
- **Meta**: Maps to "be open and set direction" — you built internal consensus, not top-down mandate
- **Pinterest**: Maps to collaborative influence — you persuaded through example and enablement
- **Roblox**: Maps to community/platform thinking — you built the tool AND drove adoption

**Watch out for:** The story falls flat if adoption is still in progress. Have a number: "X of Y teams now use this, saving ~Z hours/quarter."
