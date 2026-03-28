# Uber — Behavioral Prep

Behavioral format: behavioral + values-fit, usually 30–45 min. Probes ownership, customer obsession, and marketplace thinking.

---

## Uber's Cultural Norms (Relevant to MLE)

- **We build for and with the community**: impact is measured in trips, earnings, safety — not just model metrics
- **We make big bold bets**: taking calculated risks; being willing to be wrong on large-scope decisions
- **We persevere**: working through ambiguity, hard problems, organizational friction
- **We celebrate differences**: global user base; geographic and demographic assumptions are bugs
- **We do the right thing**: safety, privacy, regulatory compliance — non-negotiable, even under speed pressure
- **We act like owners**: treat your domain as if it's your company; no "it's not my problem"

---

## What They're Actually Testing

- **Ownership under pressure**: Uber moves fast and the business is complex — marketplaces, safety, regulation. They want to see you've taken ownership of hard problems even when it wasn't your mandate.
- **Marketplace intuition**: Do you think about second-order effects? If your model changes driver behavior, does that affect rider wait times? If your surge algorithm changes, does that affect driver earnings? Systems thinking across supply/demand.
- **Customer obsession (both sides)**: Uber serves drivers and riders. Optimizing for one at the expense of the other is a signal failure. Stories that show awareness of both sides of the marketplace are valued.
- **Operating in ambiguity**: Uber is a real-world business — things don't work the way simulations say they will. Stories where you adapted to messy real-world data/behavior are the target.
- **Speed with accountability**: Uber is fast-moving but operates in a regulated industry. The Staff answer balances urgency with rigor.

---

## Question Patterns

| Question | Cultural norm it maps to | Story to tell |
|---|---|---|
| "Tell me about a time you took ownership of a problem outside your scope." | Act like owners | `ambiguity-and-scope.md` — Template 3 (inheriting broken system) OR `cross-functional-influence.md` — Template 1 |
| "Tell me about working in a high-ambiguity environment." | Persevere | `ambiguity-and-scope.md` — Template 1 or 2 |
| "Tell me about a difficult tradeoff between two important metrics." | Big bold bets | `prioritization-and-tradeoffs.md` — Template 1 (saying no with data) |
| "Tell me about a time your model or system had unintended consequences." | Do the right thing | `failure-and-learning.md` — Template 1 (production incident) — focus on how you caught it and what changed |
| "Tell me about moving fast in a situation where mistakes were costly." | Persevere + Do the right thing | `conflict-and-disagreement.md` — Template 1 |
| "Tell me about building something that affected a real-world user experience." | Build for the community | Any story with clear end-user impact measurement |
| "Tell me about changing course based on new information." | Big bold bets | `failure-and-learning.md` — Template 2 |

---

## Uber-Specific Behavioral Angle: Marketplace Effects

Uber is a two-sided marketplace with cascading effects. Be ready for follow-up questions:

- "How did you verify that your ML change didn't hurt driver earnings?"
- "What monitoring did you set up to catch if your model was creating adverse selection?"
- "How did your model perform during surge events vs. normal conditions?"

**Strong framing**: "When I changed the ETA model, I tracked downstream effects on driver acceptance rate, trip completion rate, and rider wait time — not just prediction error. An improvement in RMSE that degrades driver acceptance rate is not actually an improvement."

---

## Signals That Differentiate Staff from Senior

- **Systems thinking**: Senior thinks about their model's direct metric. Staff thinks about the model's effect on the marketplace — supply, demand, and their interaction.
- **Real-world data messiness**: Strong Uber candidates have stories about dealing with data that doesn't match training distribution — because Uber's real world is messy (weather, events, regulatory changes).
- **Geographic awareness**: Global systems that work locally. Stories that show sensitivity to how the model behaves differently in Lagos vs. San Francisco vs. Tokyo.
- **Ownership of outcomes, not just work**: "I shipped it" is not Staff at Uber. "I shipped it, monitored it, caught the edge case in week 2, and fixed it before it became a P1" is Staff.

---

## Red Flags (Lower the Bar)

- Stories that don't connect to real user (rider/driver) impact
- "I optimized the model metric" without checking downstream marketplace effects
- No ownership of monitoring and post-launch behavior
- "We" throughout with no clear individual contribution
- Stories set entirely in offline evaluation — no production deployment experience
