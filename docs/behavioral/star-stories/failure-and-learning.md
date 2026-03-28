# Failure & Learning

P1: OpenAI (✓✓), Netflix (✓✓), Stripe (✓✓). The most revealing behavioral question at Staff level.

---

## Common Question Phrasings

- "Tell me about a project that failed."
- "Tell me about a mistake you made and what you learned."
- "Tell me about a time something went wrong in production."
- "What's your biggest professional regret?"
- "Tell me about a time your model or system caused a problem."

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "The model didn't perform as well as expected. I re-tuned it and it improved." | "We shipped a model that degraded user trust at scale before we caught it. Here's what I missed in the pre-launch review, what the monitoring gap was, and the systemic change I made so the org couldn't ship that way again." |
| "I made a mistake, I fixed it, I moved on." | "I made a mistake, I fixed it, I learned X, AND I changed the process/tooling so this class of error is prevented structurally." |
| Blames external factors or teammates | Takes full ownership of what was in their control |
| Story is safe / stakes are low | Story has real business impact: revenue lost, user trust damaged, team morale affected |

**The Staff failure story must have three parts:**
1. What failed (with stakes)
2. What I missed and why (honest diagnosis, no excuses)
3. What structurally changed (not just "I'll be more careful")

---

## Story Template 1: ML Model That Caused a Production Incident

**Situation:** A model you built or owned was deployed and caused a measurable negative outcome — degraded user experience, incorrect decisions made at scale, downstream system overloaded. You didn't catch it before it affected users.

**Task:** Contain the damage, do the RCA, and rebuild trust with the team/stakeholders.

**Action:**
1. Contain immediately: roll back the model or add an override. Don't debug in production under load.
2. Do an honest RCA (not a blame-free one): "Here are the 3 things I could have done that would have caught this: (1) adversarial test cases for the edge case that bit us, (2) shadow mode period before full rollout, (3) monitoring alert on output distribution shift."
3. Own what was in your control: "I made the call to skip shadow mode to hit the launch date. That was my call. Here's what I'd do differently."
4. Write the post-mortem: specific, named action items with owners and due dates. Not "we'll be more careful" — "we'll add distribution shift monitoring to the deploy checklist by [date]."
5. Present it transparently: walk the stakeholders through the post-mortem, including the mistakes. Transparency after a failure is more trust-building than defensiveness.

**Result:** [Fill in: how quickly was it caught? what was the scale of impact? what specific monitoring/process change was made? did it prevent a similar incident?]

**Best fit for:**
- "Tell me about a time something went wrong in production"
- "Tell me about a mistake with real consequences"

**Company fit:**
- **OpenAI**: Intellectual honesty — you named your mistakes precisely, not vaguely
- **Netflix**: "Highly aligned, loosely coupled" — you owned the RCA and the fix without requiring hand-holding
- **Stripe**: "Trust" — your post-mortem rebuilt trust with users/operators by being specific about what changed

**Watch out for:** If your failure story has no real user impact, it's not a Staff-level story. The stakes need to be high enough that the recovery required real judgment and real change.

---

## Story Template 2: Wrong Technical Bet That Wasted Quarters

**Situation:** You advocated for a technical direction — a new framework, a custom-built system, a model architecture — spent significant team resources on it, and it didn't deliver. Either the technical approach didn't work, or the use case shifted, or the bet was simply wrong.

**Task:** Recognize the mistake before it's obvious, stop the bleeding, and redirect the team.

**Action:**
1. Recognize the pattern early: "By month 2, the metrics weren't moving in the right direction. I kept explaining it away. Looking back, I should have called it at month 2, not month 5."
2. Make the stop decision explicitly: "I called a team meeting and said: 'This isn't working. I advocated for this direction, I was wrong. Here's the evidence. Here's my recommended pivot.'"
3. Own the cost honestly: "This cost 4 engineer-months. Here's what we could have built instead. Here's how we recover."
4. Change the process: "Going forward, I set explicit kill criteria at the start of every significant technical bet: 'If metric X doesn't improve by Y by date Z, we stop.' No more sunk cost reasoning."

**Result:** [Fill in: how quickly was the pivot made? what was eventually built? what's the process change that came out of it?]

**Best fit for:**
- "Tell me about a time you changed course based on evidence"
- "Tell me about a professional regret"
- "Tell me about when you were wrong"

**Company fit:**
- **Stripe**: "Operate with rigor" — you built kill criteria into future bets, not just acknowledged the mistake
- **OpenAI**: "Intellectual honesty" — you named "I was wrong" explicitly in a team setting
- **Shopify**: "Long-term thinking" — you made the short-term painful decision (stop the project) to protect long-term outcomes

**Watch out for:** "I made a mistake but it all worked out fine" is not a failure story. There needs to be real cost, real recognition of your own error in judgment, and real structural change.

---

## Story Template 3: Interpersonal Failure — Damaged a Working Relationship

**Situation:** You handled a professional relationship badly — gave feedback in the wrong forum, advocated too hard for your view in a way that damaged trust, or failed to give someone recognition they deserved. The relationship degraded and it affected team dynamics.

**Task:** Acknowledge the mistake, repair the relationship, and change the behavior that caused it.

**Action:**
1. Identify the mistake precisely: not "I was too direct" — "I gave critical feedback on X's work in a group code review rather than privately first. That was a public undermining, not a technical review."
2. Repair it directly: "I went to X directly, acknowledged what I did, didn't make excuses, and asked what I could do differently."
3. Understand the root cause: "I was under deadline pressure and skipped the 1:1 I should have had first. That's a pattern under stress — I've since added 'pre-review 1:1' to my process for sensitive feedback."
4. Don't over-correct: "I didn't try to become their best friend or overexplain. I acknowledged, changed the behavior, and let time rebuild the trust."

**Result:** [Fill in: did the relationship repair? what specifically changed in your approach? is the person still on the team / did they thrive?]

**Best fit for:**
- "Tell me about a time you made a mistake with a teammate"
- "Tell me about a time you received difficult feedback"
- "Tell me about a time you had to repair a relationship"

**Company fit:**
- **Netflix**: "We say what we think" + "we care about each other" — you had the conversation directly, acknowledged the specific mistake, didn't avoid it
- **Meta**: "Be open" — you modeled the behavior you'd want others to show

**Watch out for:** Avoid stories where you were the victim ("they were difficult to work with"). Take ownership of your contribution to the dynamic.
