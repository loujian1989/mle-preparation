# Technical Leadership

P1: Meta (✓✓), Netflix (✓✓), OpenAI (✓✓). Differentiates Staff from Senior more than any other competency.

---

## Common Question Phrasings

- "Tell me about a time you raised the technical bar on your team."
- "Describe how you've mentored or grown engineers around you."
- "Tell me about setting technical direction for a project or team."
- "Tell me about a time you had to make a critical architectural decision."
- "How do you hold the quality bar when the team is moving fast?"

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "I reviewed PRs carefully and gave good feedback." | "I identified that our code review process was creating a bottleneck and lowering quality simultaneously. I redesigned the process — defined what's required vs. optional, trained the team on it, and we cut review latency by 40% while catching 2× more production bugs." |
| "I mentored a junior engineer." | "I identified that X engineer had the skills to be the team's future tech lead. I deliberately gave them increasingly ambiguous work, coached them on stakeholder communication, and they're now leading the platform migration." |
| "I made a good architectural decision." | "I defined the decision-making process — who needs to be in the room, what level of analysis is required, when to escalate. The outcome was good, but the lasting contribution is the process." |
| Technical impact = one project | Technical impact = how the team makes decisions going forward |

---

## Story Template 1: Reversing a Technical Debt Spiral

**Situation:** A system your team owns has accumulated significant technical debt — slow deployments, frequent incidents, fear of touching legacy code. The business wants new features; the team is stuck firefighting. Leadership doesn't see the debt as a priority because it's invisible.

**Task:** Make the invisible visible and get the investment to fix it — without stopping feature work.

**Action:**
1. Quantify the cost of the status quo: "We spend X hours/sprint on incidents caused by this system. At $Y/engineer-hour, that's $Z/quarter. The system has caused N P1 incidents in the last 6 months."
2. Build the business case: translate technical debt into product risk. "If this system fails during [critical business period], the revenue impact is $X."
3. Propose a parallel path: never ask to "stop features to fix tech debt." Propose a 20% allocation model — feature dev continues at 80%, debt reduction at 20%.
4. Define done: write the target architecture doc. "Done = system can deploy in <10 minutes, has <0.1% error rate, and can be modified by any engineer on the team without fear."
5. Demonstrate progress: ship visible milestones (first service migrated, first test suite green) before asking for more investment.

**Result:** [Fill in: how much debt was reduced? what was the incident rate change? what new capabilities became possible?]

**Best fit for:**
- "Tell me about raising the technical bar"
- "Tell me about your biggest technical contribution"

**Company fit:**
- **Meta**: "Move fast AND reliably" — you addressed the debt that was actually slowing velocity
- **Shopify**: Lead with "build for the long term" — the 20% model is a sustainable practice, not a one-time fix
- **Netflix**: Lead with judgment — you made the invisible visible and convinced leadership with data, not advocacy

**Watch out for:** Don't frame this as "I fixed the tech debt." Frame it as "I changed how the team thinks about and funds technical health."

---

## Story Template 2: Defining the Technical Direction Under Uncertainty

**Situation:** The team needs to make a foundational architectural decision — a choice that will shape work for the next 12–18 months. The options are genuinely uncertain. There's no consensus, and waiting has a cost.

**Task:** As the most senior technical person, you need to make the call and own it — even without complete information.

**Action:**
1. Define the decision criteria first: before comparing options, write "here's what we need to be true about the chosen solution in 18 months: latency <X, scale to Y, supports Z use cases."
2. Time-box the research: "We'll spend 2 weeks on spikes, not 2 months. Here's what each spike will answer." Assign owners.
3. Document the alternatives explicitly: write a 1-page comparison — not a 20-page RFC. Include what each alternative is good for and what it fails at.
4. Make the call: "Based on our criteria and the spike results, I'm recommending Option A. Here's what we're giving up, and here's the trigger that would make me revisit this decision."
5. Communicate down and up: brief your team on the decision and the reasoning. Brief your manager on the risks you see. Create a decision record.

**Result:** [Fill in: what was decided? did it hold up? what would have happened with the other option?]

**Best fit for:**
- "Tell me about making a critical technical decision"
- "Tell me about technical direction-setting"

**Company fit:**
- **OpenAI**: "Intellectual honesty" — you framed the decision with explicit uncertainty and a trigger for revisiting
- **Stripe**: "Precision" — you time-boxed uncertainty resolution rather than debating indefinitely
- **Uber**: "Ownership" — you made the call without waiting for perfect information

**Watch out for:** The story needs a real decision with real stakes. "I chose library A over library B" is not Staff level. The decision should affect team direction for quarters, not weeks.

---

## Story Template 3: Growing a Senior Engineer Into a Tech Lead

**Situation:** You identify an engineer on your team who has the raw skills to become a tech lead but lacks the cross-functional and communication skills to operate at that level. They're stuck in a Senior IC loop — technically strong but not yet influential.

**Task:** Grow them into tech lead readiness without giving them a title they're not ready for — and without it taking 3 years of passive mentoring.

**Action:**
1. Diagnose the gap specifically: "Your technical skills are ready. The gap is: you explain solutions before you've confirmed you understand the problem. That's causing PMs to not trust your estimates."
2. Give deliberate stretch assignments: put them in meetings they're not ready for — but debrief afterward. "You went into that meeting to get alignment. You left without it. What happened? What would you do differently?"
3. Co-present with them: have them lead the next technical review, with you as backup. Let them stumble slightly; debrief on what they'd do differently.
4. Give them a real scoped project: a project with a PM, cross-functional stakeholders, and a real deadline — where they're the technical owner and you're the advisor.
5. Advocate for them openly: name them in leadership reviews. "X is the technical owner of this project." Create visibility so their work is attributed.

**Result:** [Fill in: did they get promoted? what project did they successfully lead? what's the team's capacity now?]

**Best fit for:**
- "Tell me about mentoring or growing engineers"
- "Tell me about your investment in team development"
- "Tell me about a time you made your team stronger"

**Company fit:**
- **Netflix**: "Keeper test" framing — you're building the team that makes you replaceable, which is the highest-leverage thing you can do
- **Meta**: "Impact through others" — at Staff level, your impact multiplies through the people you develop
- **Roblox**: "Community" — you invested in building the team's capability, not just delivering features yourself

**Watch out for:** Don't present this as "I was a great mentor and they loved working with me." Present it as a deliberate, structured intervention with a specific outcome: they grew into a specific capability you identified as missing.
