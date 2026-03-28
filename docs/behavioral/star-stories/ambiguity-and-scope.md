# Ambiguity & Scope

P1: OpenAI (✓✓), Shopify (✓✓), Uber (✓✓), Whatnot (✓✓). Core Staff competency: defining the problem, not just solving it.

---

## Common Question Phrasings

- "Tell me about a time you worked on a project with unclear requirements."
- "Tell me about a time you had to define scope from scratch."
- "Describe a situation where the problem wasn't well-defined when you started."
- "Tell me about taking ownership of something that was ambiguous."
- "Tell me about building something where nobody had done it before."

---

## Staff-Level Differentiators

| Mid-level answer | Staff answer |
|---|---|
| "The requirements were unclear so I asked my manager for more detail." | "There were no requirements. I wrote the first draft, got alignment from 3 stakeholders, and defined the success criteria before anyone had thought about them." |
| "I figured it out as I went." | "I identified the 3 core uncertainties, designed the minimum experiments to resolve each, and made explicit decisions at each gate — documented so the team could continue without me." |
| Story = technical ambiguity | Story = strategic ambiguity: which problem to solve, what success means, whether to build at all |
| Resolved ambiguity for themselves | Resolved ambiguity for the whole team/org — leaving behind a framework others can use |

---

## Story Template 1: Zero-to-One ML Project With No Prior Art

**Situation:** You're asked to "build an ML system for X" where X has no prior ML solution at your company, no established dataset, no clear success metric, and no one who has done it before. Leadership wants it. No one knows what "done" means.

**Task:** Go from "build something for X" to a shipped, evaluated system — entirely through your own judgment about what to build.

**Action:**
1. Start with the business problem, not the ML problem: "What decision does this system enable? What does a human do today, and how does ML make it better/faster/cheaper?"
2. Define success before touching data: "I'll propose success criteria to my PM and get sign-off before building anything. If we can't define done, we shouldn't start."
3. Scope to the smallest useful version: "Instead of 'build the full system,' I proposed: 'Build a prototype that answers the single most important sub-question. Ship that, learn, then scope the real system.'"
4. Document every assumption: "I wrote a 1-page project brief with the assumptions we're making. Every assumption is a potential failure point."
5. Set explicit go/no-go criteria: "If we reach milestone X and the offline metric isn't Y, we re-evaluate whether this is the right approach."

**Result:** [Fill in: what got shipped? what was the most important pivot made after the initial prototype? what's still running in production?]

**Best fit for:**
- "Tell me about taking initiative on an ambiguous project"
- "Tell me about zero-to-one work"

**Company fit:**
- **OpenAI**: "Curiosity and first-principles thinking" — you started by questioning whether the stated problem was the right problem
- **Whatnot**: "Founder mentality" — you operated like an owner, defining what needed to be built before building it
- **Shopify**: "Why before how" — you explicitly validated the business case before writing code

**Watch out for:** The Staff failure mode here is jumping to implementation. If your answer focuses mostly on what you built technically, you've missed the ambiguity-resolution piece.

---

## Story Template 2: Scope Creep on a Multi-Quarter Project

**Situation:** A project that started as a well-scoped 2-quarter effort has grown through stakeholder additions, new requirements from leadership, and technical discoveries. It's now threatening to take 6+ quarters without a clear end state. Team morale is declining.

**Task:** Stop the scope spiral without dropping stakeholder relationships or killing the project.

**Action:**
1. Diagnose what's in vs. out: draw a hard line between "must-have for the original promise" and "nice-to-have that was added." Get stakeholders to agree in writing.
2. Propose a tiered release: "Here's what ships in Q+1 (core value), here's what ships in Q+2 (first extension), here's what goes on the backlog (future)."
3. Make the cost of scope addition visible: "Every time someone adds to this project, I quantify the cost: this request adds 3 weeks and pushes the launch date to [date]. Do we still want it?"
4. Get explicit re-authorization from leadership: "I brought the full scope to my VP and said: 'This has grown from a 2-quarter project to a 6-quarter project. I need explicit approval to continue at this scope, or direction to cut it back to the original plan.'"

**Result:** [Fill in: what was cut? what shipped on the reduced scope? what was the team's experience after the scope cut?]

**Best fit for:**
- "Tell me about navigating changing requirements"
- "Tell me about a time you said no or pushed back on scope"

**Company fit:**
- **Shopify**: "Do less better" — you explicitly cut scope to ship something excellent, not everything adequately
- **Reddit**: "Ruthless prioritization" — you made the tradeoff visible and got explicit agreement
- **Uber**: "Ownership" — you brought the scope problem to leadership rather than silently struggling with it

**Watch out for:** "We just did all the scope" is not this story. The Staff version requires having said no to something, or forcing a choice that others were avoiding.

---

## Story Template 3: Inheriting a System With No Documentation

**Situation:** You inherit ownership of a production system — ML model, pipeline, or service — that was built by someone who has left the company. No documentation. No test coverage. It's business-critical. Something about it is broken or will break soon.

**Task:** Get to understanding and control of this system without taking it down, while also doing your regular work.

**Action:**
1. Do an emergency documentation pass: spend 2 days writing down everything you discover — "how it works now" even if you don't know why. Create a shared doc.
2. Instrument before touching: add logging and monitoring before you change anything. You need a baseline to know if your changes broke something.
3. Identify the highest-risk unknowns: "What are the 3 things that could cause a P1 if they're wrong? Those get investigated first."
4. Build a test harness before fixing: write regression tests that capture current behavior (even if that behavior is wrong). They protect you during refactoring.
5. Communicate proactively: "I told my manager: 'This system has X critical gaps. My plan to address them in Y timeline. The risks in the interim are Z.'"

**Result:** [Fill in: how long until you had full working knowledge? what did you find that was broken? what process change prevents this from happening to the next person?]

**Best fit for:**
- "Tell me about operating under uncertainty"
- "Tell me about taking ownership of something you didn't create"

**Company fit:**
- **Meta**: "Move fast AND reliably" — you didn't just accept the risk; you instrumented and documented before it became a P1
- **Netflix**: "Context not control" — you built the context before taking action; you didn't make changes from ignorance
- **Stripe**: "Rigor" — systematic approach under pressure; you didn't guess

**Watch out for:** "I eventually figured it out" is not a Staff story. The Staff version shows a deliberate, methodical approach with explicit risk communication and a process that leaves the system better for the next person.
