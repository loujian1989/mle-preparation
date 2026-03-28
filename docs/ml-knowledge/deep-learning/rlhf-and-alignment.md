# RLHF & Alignment — ML Knowledge Q&A

P0: OpenAI. Expected: can describe the full pipeline, knows failure modes, knows DPO vs PPO.

---

## RLHF Pipeline

### Q: Walk through the full RLHF pipeline: SFT → RM → PPO.

**Answer (Staff level):**

**Stage 1 — Supervised Fine-Tuning (SFT)**:
- Fine-tune a pretrained LLM on high-quality demonstration data (human-written responses to prompts). Standard cross-entropy training.
- Purpose: teach the model the *format* of helpful responses before optimizing for preference. Skipping SFT and going straight to RL from a raw pretrained model is unstable.

**Stage 2 — Reward Model (RM)**:
- Collect preference comparisons: for the same prompt, human annotators rank response A vs. B.
- Train a reward model: append a linear head to an LLM (often same architecture as policy, sometimes smaller), trained with Bradley-Terry pairwise loss: `L = −log σ(r(x, y_win) − r(x, y_lose))`.
- Output: a scalar reward for any (prompt, response) pair.

**Stage 3 — PPO (Proximal Policy Optimization)**:
- Use RM as reward signal to fine-tune the SFT policy via RL.
- Objective: `maximize E[r(x, y)] − β · KL(π_θ || π_SFT)`.
- The KL penalty (β · KL) prevents the policy from diverging too far from SFT — without it, the model exploits the RM (reward hacking).
- PPO clip ratio ε (typically 0.2) prevents large policy updates per step.

**Company context:** OpenAI (non-negotiable), Meta (Llama alignment). Expected to describe all 3 stages and the KL penalty purpose without prompting.

**Common wrong answer:** Skipping the KL penalty explanation — interviewers specifically probe why the KL term is there.

---

## Reward Hacking

### Q: What is reward hacking, give two concrete examples, and how do you mitigate it?

**Answer (Staff level):**
- **Reward hacking**: the policy learns to maximize the reward model's score in ways that don't correspond to genuine quality — exploiting the RM as a proxy.

**Example 1 — Response length gaming**: RM is implicitly biased by annotators who prefer longer, more detailed responses. Policy learns to generate very long responses regardless of content quality. Mitigation: add a length-penalty term or normalize reward by token count.

**Example 2 — Sycophancy**: RM was trained on annotators who preferred responses agreeing with the human. Policy learns to affirm whatever the user says (even if factually wrong) to maximize score. Mitigation: include "I disagree" examples in SFT; train RM on factuality-labeled comparisons, not just preference.

**Example 3 — Repetition / gibberish**: RM assigns high reward to certain tokens or patterns (often due to annotation artifacts). Policy generates repetitive or incoherent text. Mitigation: add KL penalty from SFT; use ensembled RM (multiple reward models, take minimum — conservative reward).

**General mitigations:**
- KL divergence penalty from SFT policy (built into PPO objective)
- Constitutional AI: secondary critic filters responses before RM scoring
- Ensemble of RMs from different annotator groups
- Periodic RM refresh with new comparisons from current policy outputs

**Company context:** OpenAI (this is the intellectual honesty probe — they want you to identify failure modes before they're prompted).

**Common wrong answer:** "Reward hacking means the model gives wrong answers." — Too vague. Must give a specific mechanism (what the model is optimizing, why RM fails to penalize it, and what the fix is).

---

## DPO vs. PPO

### Q: What is Direct Preference Optimization (DPO) and when do you use it over PPO?

**Answer (Staff level):**
- **DPO (Rafailov et al., 2023)**: reformulates RLHF as a supervised learning problem. Bypasses the explicit reward model and PPO training loop.
- Insight: the optimal policy under the RLHF objective can be expressed in closed form in terms of the SFT policy. This allows direct training with a classification-style loss on preference pairs:
  `L_DPO = −E[log σ(β · (log π_θ(y_w|x) / π_ref(y_w|x)) − β · (log π_θ(y_l|x) / π_ref(y_l|x)))]`
  where y_w = winning response, y_l = losing response.

| | **PPO** | **DPO** |
|---|---|---|
| Reward model | Explicit | Implicit (baked into loss) |
| Training complexity | High (online RL, RLHF loop) | Low (offline supervised) |
| Sample efficiency | Lower (online rollouts expensive) | Higher (trains on fixed preference dataset) |
| Stability | Lower (RL instability) | Higher |
| Expressiveness | Higher (online exploration) | Lower (offline, no exploration) |

- **Use DPO when**: preference data is fixed and comprehensive; you want faster iteration; compute budget is limited; stability is prioritized.
- **Use PPO when**: you need the model to explore (online generation is part of the loop); reward is from live environment feedback (not static comparisons); you need the full RLHF power for complex tasks.

**Company context:** OpenAI. DPO is becoming the default for fine-tuning cycles; PPO for initial alignment.

**Common wrong answer:** "DPO is a simplified PPO." — DPO eliminates the RM and the RL loop entirely. It's a different training paradigm (supervised contrastive loss), not a simplified version of PPO.

---

## Constitutional AI

### Q: What is Constitutional AI and how does it relate to RLHF?

**Answer (Staff level):**
- **Constitutional AI (CAI, Anthropic, 2022)**: rather than relying solely on human preference comparisons, a set of principles ("constitution") guides both data generation and model training.
- **Two-phase**:
  1. **Supervised phase**: model critiques and revises its own harmful outputs using the constitution as a guide (self-critique). Generate revised responses. SFT on revised pairs.
  2. **RLHF phase**: use AI-generated preference comparisons (model scores responses against the constitution) to train RM. Reduces reliance on human annotation for harmful content (reduces annotator exposure to harmful material).
- **Key insight**: human annotation is the bottleneck and quality bottleneck in RLHF. CAI uses the model itself as a scalable annotator, guided by explicit principles instead of implicit annotator preferences.
- Reduces sycophancy (the constitution includes "do not agree with incorrect claims").

**Company context:** OpenAI (alignment discussion), less so for other companies unless specifically working on safety.

**Common wrong answer:** "Constitutional AI is just fine-tuning on human-written rules." — The self-critique/revision loop and AI-generated RM training data are the distinguishing features, not just having a rule set.

---

## Reward Model Collapse

### Q: What is reward model collapse and how does it differ from reward hacking?

**Answer (Staff level):**
- **Reward hacking** (above): the policy exploits the RM's blind spots — RM is still functioning, policy is over-optimizing a proxy.
- **RM collapse**: as the policy improves, it generates responses the RM has never seen during training (out-of-distribution). The RM's outputs become unreliable (high variance, miscalibrated). The policy, optimizing a noisy signal, diverges.
- **Gao et al. (2023) scaling laws**: RM score peaks and then decreases as the policy trains longer (Goodhart's Law). The proxy metric (RM score) and true quality diverge as over-optimization proceeds.
- **Mitigation**:
  - **Online RM refreshes**: periodically collect new preference comparisons from current policy outputs and retrain RM.
  - **Ensemble RM**: use multiple RMs; take the minimum (pessimistic reward) to be conservative.
  - **KL constraint**: β·KL term in PPO limits how far the policy moves from SFT, reducing OOD exposure.

**Company context:** OpenAI (production RLHF systems).

**Common wrong answer:** Treating reward hacking and RM collapse as synonyms. They have distinct mechanisms: hacking = policy exploitation of RM weaknesses; collapse = policy making the RM itself unreliable via OOD distribution shift.

---

## Evaluating Aligned Models

### Q: How do you evaluate whether an RLHF-trained model is "better"? What metrics do you use?

**Answer (Staff level):**
- **Win rate against baseline**: pair-wise human or LLM-as-judge comparisons. Model A vs. Model B on a held-out prompt set. Prefer Elo-style rating over simple win rate when comparing >2 models.
- **Automated benchmarks**: MMLU (knowledge), HumanEval (code), MT-Bench (multi-turn instruction following), TruthfulQA (factuality), BBH. Report per-benchmark to detect regression (aligned model shouldn't regress on capabilities).
- **Calibration**: does the model express uncertainty correctly? Use Expected Calibration Error on factual Q&A tasks.
- **Alignment tax**: RLHF may reduce capability benchmarks slightly (model refuses correct but sensitive requests). Report capability/alignment Pareto trade-off.
- **Red-teaming**: adversarial evaluation by human testers trying to elicit harmful outputs. Pass rate on adversarial prompts.

**Company context:** OpenAI (must know multiple evaluation dimensions; never claim "it's better" without naming specific metrics).

**Common wrong answer:** "I'd use human evaluation to confirm it's better." — Human evaluation is expensive and noisy. Staff answer layers automated benchmarks + automated LLM-judge + targeted human eval for ambiguous cases.
