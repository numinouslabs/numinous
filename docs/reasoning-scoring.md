# Reasoning Scoring

Miners can submit free-form reasoning alongside their numeric prediction for an event. That reasoning is evaluated through **two independent scores**, which are then combined into a single reasoning score.

A **substantial share of daily emissions is allocated to the Reasoning pool** — currently **25% on the Information track** and **20% on the Signals track** (visible under "Current Pool alphas" on the [leaderboard](https://leaderboard.numinouslabs.io/)). On the leaderboard's per-miner table, the **`Reasoning` column is exactly the weighted average defined below** — i.e., `0.70 * rubric quality + 0.30 * implied-probability Brier`. Click into your hotkey to inspect the rubric breakdown, the underlying reasoning text, and the per-event scores that fed into it.

> **Length limit:** Reasoning is currently **truncated at 2,500 characters** before scoring. Anything past that cutoff is discarded by both the rubric judge and the implied-probability extractor — write tight, front-load your strongest evidence and citations.

## The two component scores

### 1. Reasoning-quality rubric score (70%)

The reasoning is graded by an LLM judge against a five-dimension rubric (each dimension scored 1–5). This measures the *quality* of the argument itself — independent of whether its implied probability happened to land near the outcome.

The five dimensions are:

1. **Sources Used** — citations, attribution, reliability, source-quality discrimination, and the link between cited sources and the rationale.
2. **Evidence Extracted** — relevance, depth, ranking of importance, and absence of meaningful omissions.
3. **Combination & Weighting** — explicitness, transparency, and rigor of how evidence is combined into a conclusion.
4. **Uncertainties / Counterpoints** — identification, depth, and quantified impact of relevant uncertainties and opposing views.
5. **Mapping to Final Probabilities** — traceability from evidence and weights to the actual probabilities reported.

Full rubric below.

### 2. Implied-probability Brier score (30%)

The reasoning text is also fed to a separate LLM extractor that derives the **probability the analysis itself implies** — independent of any number the miner stated. The exact extractor instruction is:

> **implied_probability**: Extract what probability the reasoning's analysis implies. Derive it from the evidence and logic, not from bare stated numbers. If the reasoning shows analytical work leading to a number (modeling, base rates, threshold analysis), use what the analysis supports. If there is NO analysis, default to 0.5.

That extracted probability is then scored against the resolved outcome with a standard Brier score. This rewards reasoning whose internal logic actually points at the eventual outcome — not reasoning that quotes a confident number with no support.

Reasoning containing no real analysis collapses to the 0.5 default, which is roughly the worst Brier score for a binary event resolved at 0 or 1.

## Final reasoning score

```
reasoning_score = 0.70 * quality_rubric_score
                + 0.30 * brier_from_implied_probability
```

The 70/30 split puts most of the weight on **the structural quality of the argument** (sourcing, evidence, transparent weighting, uncertainty handling, traceable probabilities), while still meaningfully rewarding reasoning whose implied probability calibrates to the resolved outcome.

## Rubric reference

The judge LLM uses the following rubric. Scores are integers 1–5 per dimension.

### 1. Sources Used (Citations, Attribution & Reliability)

- **5 — Exceptional:** Every fact tied to a *direct, authoritative* source. Sources are **high-reliability** (primary government data, central bank reports, peer-reviewed research, official statements) and pulled from the list of sources provided to the predictor. Sources weighted by reliability with clear recognition that primary, authoritative sources like Fed/Treasury/BLS data > news from reputable sources > provided market data >> articles >> blogs. Zero broken links, zero vague attributions. Connection between rationale and sources is very clear.
- **4 — Good:** All major claims properly sourced with mostly high-reliability sources, but **exactly one minor flaw** (e.g., one secondary source where primary was available, or one minor formatting issue). Still shows clear source quality discrimination. Connection between rationale and sources is clear, but not explicit.
- **3 — Adequate:** Most important claims sourced, but **multiple significant weaknesses**: broken links, 2–3 lower-quality sources treated as authoritative, or poor source quality discrimination. Mix of reliable and unreliable sources without proper weighting. Connection between rationale and sources is implied and not completely clear.
- **2 — Poor:** Sourcing is fundamentally inadequate. Either most claims lack direct sources, OR heavy reliance on weak sources (news summaries, blogs, non-specialist outlets), OR no recognition of source quality differences. Connection between rationale and sources is unclear; cited sources don't appear to have meaningfully impacted the rationale and prediction.
- **1 — Terrible:** No meaningful citations, only unreliable sources, or completely broken/fabricated references. Connection between rationale and sources is completely unclear.

### 2. Evidence Extracted (Relevance & Ranking)

- **5 — Exceptional:** Extracts *every* critical piece of evidence with surgical precision. Goes far beyond surface-level to uncover deeper insights. Perfect ranking of importance. Demonstrates comprehensive understanding of what drives the outcome. Zero meaningful omissions.
- **4 — Good:** Extracts most critical evidence with good depth, but **misses exactly one important element** or slightly misranks importance. Generally goes beyond surface-level with meaningful insights.
- **3 — Adequate:** Extracts reasonable evidence but with **noticeable gaps or shallow treatment**. Some insights beyond headlines, but several areas lack depth or miss key components that should influence predictions.
- **2 — Poor:** Evidence is mostly superficial headline-level facts. Limited insight into underlying drivers. Significant omissions of relevant information.
- **1 — Terrible:** No meaningful evidence extraction. Only surface-level or irrelevant facts that provide no predictive insight.

### 3. Combination & Weighting (Reasoning Transparency)

- **5 — Exceptional:** Crystal clear step-by-step reasoning with **explicit numerical weights** and rock-solid justification for each weight. Complete transparency in how evidence combines. Mathematical/logical rigor throughout.
- **4 — Good:** Reasoning mostly explicit with clear evidence combination, but **weights are somewhat implicit** or justification could be slightly more rigorous.
- **3 — Adequate:** Basic combination logic present but **lacks precision or depth**. Weighting is implied rather than explicit, or reasoning has logical gaps.
- **2 — Poor:** Minimal attempt at systematic combination. Mostly just lists evidence without clear integration logic.
- **1 — Terrible:** No discernible combination methodology. Pure list of facts with no integration.

### 4. Uncertainties / Counterpoints (Balance & Awareness)

- **5 — Exceptional:** Identifies and **deeply explores multiple specific uncertainties** with quantified impact on probabilities. Shows sophisticated understanding of how different types of uncertainty (data, model, implementation, external factors) interact and compound.
- **4 — Good:** Identifies relevant uncertainties with reasonable depth, but **exploration is somewhat surface-level** or impact on probabilities not fully quantified.
- **3 — Adequate:** Acknowledges uncertainty, but treatment is **generic or superficial**. Limited exploration of how uncertainties affect the prediction.
- **2 — Poor:** Minimal acknowledgment of uncertainty. Vague statements without substance.
- **1 — Terrible:** No meaningful recognition of uncertainty or completely one-sided analysis.

### 5. Mapping to Final Probabilities (Traceability)

- **5 — Exceptional:** Every probability is **mathematically derivable** from the evidence and weights. Complete audit trail from data → logic → numbers. No probability feels arbitrary or unjustified.
- **4 — Good:** Probabilities mostly well-justified, but **1–2 numbers feel slightly under-explained** or could use more explicit derivation.
- **3 — Adequate:** Partial traceability. Some probabilities clearly derived, others feel **somewhat arbitrary or loosely connected** to evidence.
- **2 — Poor:** Probabilities appear largely disconnected from evidence. Minimal justification for the numbers.
- **1 — Terrible:** Completely arbitrary numbers with no connection to analysis.

### Scoring philosophy

- **No participation trophies** — weak work deserves low scores regardless of effort.
- **Be specific in justification** — explain *why* a score was given, with examples where applicable.

## Practical implications for miners

- **Stay under 2,500 characters.** Anything past that is dropped before scoring, so a brilliant conclusion buried at character 3,000 doesn't exist as far as either evaluator is concerned.
- The rubric is the dominant component — **structure matters more than calibration**. Cite authoritative sources, extract evidence with depth, weight it transparently, address counterpoints, and trace your probabilities back to the evidence. That's what moves 70% of the score.
- Submitting a confident number with no analysis defaults the implied probability to 0.5, dragging down the 30% Brier component. Always show your work.
- The implied-probability extractor reads *the analysis*, not the headline number. If your analysis points at 0.3 but you state 0.7, the Brier component will be scored against 0.3.
- Inspect your own reasoning scores per event from the **Miner Console** on the leaderboard — the `Reasoning` column there is this exact 70/30 weighted average.
