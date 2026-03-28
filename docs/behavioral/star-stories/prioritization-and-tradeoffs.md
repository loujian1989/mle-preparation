# Prioritization & Tradeoffs

P1: Shopify (✓✓), Reddit (✓✓), Uber (✓✓), Roblox (✓✓), Whatnot (✓✓). Core Staff competency: saying no effectively.

---

## Common Question Phrasings

- "How do you prioritize when you have competing demands?"
- "Tell me about a time you had to make a difficult tradeoff between two important things."
- "Tell me about a time you said no or pushed back on a request."
- "How do you decide what NOT to work on?"
- "Tell me about a time you had to cut scope or deprioritize something important."

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "I made a priority list and worked through it." | "I defined the prioritization framework — not just for my work, but for the team. I made the tradeoff criteria explicit so the team could make decisions without me." |
| "I said no to a stakeholder request." | "I said no with data: 'Doing this instead of X costs us $Y in expected impact. Here's the decision tree I'm using. If you want to change the decision, here's what needs to change in our strategy.'" |
| Prioritizes based on loudest voice | Prioritizes based on explicit model: expected value, reversibility, strategic alignment |
| Cuts scope to meet deadlines | Cuts scope as a deliberate quality investment — "less, done excellently" |

---

## Story Template 1: Saying No to Leadership With Data

**Situation:** Leadership or a PM makes a request that, if you do it, will delay or kill a higher-impact project. The request is reasonable on its surface. But you believe it's the wrong tradeoff given the team's current priorities.

**Task:** Push back without being obstructionist, and without just escalating to your manager to handle it.

**Action:**
1. Quantify the cost of the yes: "If I do this, here's what doesn't get done: [Project X]. The expected value of Project X is $Y based on [metric]. This request's expected value is $Z."
2. Name the decision you're making: "I'm not refusing — I'm offering a prioritized choice. Would you like me to swap this for X? If so, we need to align on which stakeholders are informed."
3. Offer a scoped alternative: "I can do a lightweight version of this in 1 week instead of 6. It won't have [feature A, B, C], but it addresses the core need."
4. Get the decision in writing: "I'm going to send a brief summary of what we discussed so there's a shared record of the tradeoff decision."
5. If they insist: "I disagree, but I'll commit. Here's my revised plan — I'm explicitly flagging that Project X is now at risk."

**Result:** [Fill in: what was the outcome? did the priority shift happen? did Project X survive? what was the relationship impact?]

**Best fit for:**
- "Tell me about saying no"
- "Tell me about prioritization with competing stakeholders"

**Company fit:**
- **Reddit**: "Do less but better" — you said no specifically to preserve quality on the higher-impact thing
- **Shopify**: "Long-term thinking" — you protected the investment that would compound, not the one that was urgent
- **Stripe**: "Operate with rigor" — your no came with an explicit expected-value analysis, not a gut feeling

**Watch out for:** If you always said yes, this is not your story. The Staff version requires an actual no (or a hard scope cut), and a willingness to own the friction that comes with it.

---

## Story Template 2: Cutting a Feature Under Launch Pressure

**Situation:** You're 2 weeks from a launch date. The project is behind. You can ship everything at lower quality, or ship a subset at high quality. Leadership wants everything; the team wants more time; users will notice both the quality gaps and any delays.

**Task:** Make the call, defend it, and execute without team morale damage.

**Action:**
1. Define what "done" means for each feature: not "finished" vs. "unfinished" — "done" means "ships with acceptable user experience." Score each feature against this.
2. Identify the MVP: "What is the minimum set of features that delivers real value to users? Everything else is a later release."
3. Communicate the cut proactively: don't wait until launch week. "I'm telling you now: Feature B is not making this launch. Here's why, here's the impact, here's when it ships next."
4. Make the tradeoff explicit to leadership: "We can ship everything at 70% quality, or ship 60% of features at 100% quality. I recommend option 2. Here's my reasoning."
5. Make the cut visible in the team: "Here's the updated scope doc. I've moved Features B and C to the next sprint. This was my call, not the team's failure."

**Result:** [Fill in: what shipped? what was the user/stakeholder reaction? did the cut feature eventually ship? what's the process going forward?]

**Best fit for:**
- "Tell me about making a difficult tradeoff"
- "Tell me about shipping under pressure"

**Company fit:**
- **Whatnot**: "Move fast and with quality" — you made the cut decisively rather than shipping something bad
- **Shopify**: "Do less better" is explicitly their culture — this story is perfect framing
- **Uber**: "Customer obsession" — the cut was made to protect user experience, not to make the team's life easier

**Watch out for:** The Staff failure mode is "the cut was forced by leadership, not chosen by me." The story needs you making the call — recommending the cut, defending it, and owning the outcome.

---

## Story Template 3: Deprioritizing Tech Debt Under Feature Pressure

**Situation:** Your team has a growing backlog of technical debt — test coverage, system observability, documentation, infrastructure upgrades. Product wants features. The debt is accumulating. One day, it will cause a major incident. But "one day" isn't a deadline.

**Task:** Protect time for technical health investment without stopping feature delivery.

**Action:**
1. Make the debt visible: create a "technical health dashboard" — incident count, deploy frequency, test failure rate, coverage%. Review it in sprint planning so it's not invisible.
2. Quantify the risk: "At current trajectory, our incident rate will double by Q+2. That's a 20% engineering overhead increase."
3. Propose a structural allocation: "I'm proposing we reserve 20% of capacity for technical health each sprint. This is not negotiable per-sprint — it's a standing allocation."
4. Choose the debt to pay first: use a simple model: "Probability of causing an incident × cost of that incident / effort to fix." Fix the highest-risk items first.
5. Show ROI: after 2 sprints, report: "Technical debt work this month: 3 engineer-weeks. Prevented incidents: estimated 2 P2s (~5 engineer-days to resolve). Net: 7 engineer-days saved."

**Result:** [Fill in: what's the incident rate now vs. before? did leadership keep the 20% allocation? what's the team culture around technical health now?]

**Best fit for:**
- "How do you balance feature work vs. technical debt?"
- "Tell me about prioritization under competing demands"

**Company fit:**
- **Roblox**: "Reliability matters for creator trust" — you protected the reliability work that protects creators
- **Meta**: "Move fast reliably" — the 20% allocation is how you move fast without accumulating uncontrolled risk
- **Reddit**: "Community trust" — the incidents you prevented would have degraded user experience

**Watch out for:** If you "always found time for tech debt somehow," that's not the story. The Staff version requires making the allocation explicit, defending it against pressure, and showing ROI.
