# Stripe — Behavioral Prep

Behavioral format: behavioral screen early in the process + embedded behavioral questions in onsite technical rounds. Stripe's bar is high for written communication and precision.

---

## Stripe's Operating Principles (Relevant to MLE)

- **Users are everything**: decisions trace back to serving users — merchants, developers, and the broader financial ecosystem
- **Operate with rigor**: precision matters; "approximately right" is not acceptable where exactness is possible
- **Move with urgency**: speed is valued, but not recklessness — urgency ≠ chaos
- **Think rigorously**: approach problems from first principles; be skeptical of cargo-culted solutions
- **Trust and integrity**: Stripe handles money and user trust is the foundation of everything
- **Global by default**: systems must work for users worldwide; geographic assumptions are bugs

---

## What They're Actually Testing

- **Precision and rigor**: Stripe is skeptical of vague answers. "It improved our metrics" will prompt "which metrics, by how much, over what time period?" Be exact.
- **User-centricity**: Every behavioral story should connect back to user impact — merchant experience, developer trust, or financial inclusion. Stories about internal efficiency without user benefit don't land.
- **Trust thinking**: Stripe is acutely aware of trust — money, fraud, compliance. Stories where you protected user trust or caught a trust-damaging risk before it hit users are highly valued.
- **Written clarity**: Stripe is known for written culture. Be prepared to be asked: "How did you communicate this to stakeholders?" — and for the process of the communication (the doc you wrote, the framework you used) to be evaluated.
- **Failure ownership**: Stripe is a high-trust environment. They want to see that you own mistakes completely, not partially.

---

## Question Patterns

| Question | Principle it maps to | Story to tell |
|---|---|---|
| "Tell me about a time a system you built affected users negatively." | Trust + Rigor | `failure-and-learning.md` — Template 1 (production incident) — go deep on RCA precision |
| "Tell me about making a judgment call that affected user trust." | Users are everything | `conflict-and-disagreement.md` — Template 1 or 3 |
| "Tell me about a time you caught a problem before it reached users." | Rigor + Trust | `ambiguity-and-scope.md` — Template 3 (inheriting broken system) OR any story where you caught a bug pre-launch |
| "Tell me about a difficult prioritization decision." | Move with urgency | `prioritization-and-tradeoffs.md` — Template 1 |
| "Tell me about setting a quality bar for your team." | Operate with rigor | `technical-leadership.md` — Template 1 or 2 |
| "Tell me about a project that required significant precision." | Think rigorously | Custom story — feature engineering, calibration, fraud modeling |
| "Tell me about a failure you caused." | Trust + integrity | `failure-and-learning.md` — Template 1 or 2 |

---

## Stripe-Specific Behavioral Angle: Fraud and Trust

If you've worked on fraud, trust, or safety systems, Stripe will go deep here. Be ready for:

- "How did you think about the false positive cost to legitimate users?"
- "How did you balance detection rate vs. user friction?"
- "What monitoring did you build so you'd know if the model was hurting legitimate users?"

**Strong framing**: "We treated false positives as a product defect, not an acceptable ML error. We tracked decline rates for legitimate users by country, card type, and merchant category. Any anomaly triggered a review."

---

## Signals That Differentiate Staff from Senior

- **Numerical precision**: Senior says "it improved." Staff says "it improved AUPRC from 0.82 to 0.89, reducing false positives by 15% while maintaining recall at 94%."
- **User impact chain**: Every decision is connected to user impact — not just "the model was better," but "the model was better in a way that reduced false declines for small merchants in Southeast Asia by X%."
- **RCA depth**: Stripe values thorough post-mortems. The staff failure story has a 5-why RCA, not "we found the bug and fixed it."
- **Process improvement**: Every failure story ends with a structural change, not a one-time fix. "I'll be more careful" is not a Stripe answer.

---

## Red Flags (Lower the Bar)

- Vague outcomes — "things improved" without numbers
- No user-centric framing — "we improved latency" without "which reduced friction for X% of checkout flows"
- Failure stories where you're a bystander — Stripe expects ownership
- "It went fine" — no tension, no hard decision, no real stakes
- Sloppy communication — Stripe probes writing and precision; sloppy verbal framing signals sloppy thinking
