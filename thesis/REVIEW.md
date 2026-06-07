# Thesis review — 3-hour deadline pass

Reviewed against the DTU report rules (Mørup 02466) and your supervisor's final-meeting notes. No broken cross-references, no undefined citations, no duplicate labels — the bibliography and `\cref` machinery are all sound. The issues below are content/consistency/prose.

**I already applied the unambiguous mechanical fixes directly** (see "Done" list at bottom — check the git diff). Everything in **Act on these** is a judgment call left for you.

---

## Act on these (highest value first)

### 1. One-tailed t-test direction contradicts itself — MAJOR (stats examiner will catch this)
- **Methodology** (`03_Methodology.tex:322`): "the alternative hypothesis was that **SAM would perform significantly better** than the best classical baseline".
- **Results** (`04_Results_and_Discussion.tex:200-201`): H1 is `mu_d > 0` with `d_i = Dice_Gaussian - Dice_SAM`, i.e. the alternative tested is **Gaussian better than SAM** — the opposite tail.

As written it reads as if you picked the tail *after* seeing SAM lose (HARKing). Two clean options:
- **Safest:** report a **two-tailed** paired t-test in both places. Your effect is large (t≈3.74), so it stays significant (two-tailed p≈9e-4 < 0.05) and you sidestep the whole objection.
- Or keep one-tailed but state the alternative as "the classical baseline outperforms SAM" in *both* sections and drop the "SAM would perform better" framing at 03:322.

### 2. Figure caption / image mismatch — MAJOR (factual)
`04_Results_and_Discussion.tex:106` — caption of the ICL comparison figure says *"based on the segmentation of **PH2** (easy)"*, but every subfigure loads **HT7a** images (`raw/easy/HT7a.jpg`, `HT7a-FATESAM2D.jpg`, etc., lines 79-104). The neighbouring classical figure (`:167`) correctly says PH2 and uses PH2 images. Decide which is right and align — almost certainly change **PH2 → HT7a** at line 106.

### 3. Image-count inconsistency: 45 vs 27 — verify
`appendices/AnAppendix.tex:33`: "As all **45** images could not be drawn all at once…". Everywhere else the test set is **27** and the case studies use **3** (`02_Data.tex:14`, `03_Methodology.tex:429`, `04:5`, `05:10`). Where does 45 come from? If it's a slip, fix it; if it refers to a real larger pool, say what it is so the reader isn't confused.

### 4. Orphan subsection breaks the numbering — structural
`03_Methodology.tex:6` — `\subsection{Development environment and tools}` appears **before the chapter's first `\section`**. It will number as "3.0.1" (or float oddly) in the ToC. Promote it to `\section{Development environment and tools}` (or fold it under the first section).

### 5. RQ1 restatement isn't verbatim — minor consistency
`03_Methodology.tex:15` paraphrases RQ1 ("How does the initial segmentation quality of SAM compare…"), whereas the RQ2 (`:330`) and RQ3 (`:442`) methodology sections quote their research questions **verbatim**. Quote RQ1 verbatim too, for parallelism, since the Conclusion checklist wants the RQs restated faithfully.

### 6. Spaced en-dash style — your own rule, 9 spots in the Introduction
You use a bare ` - ` for parenthetical dashes; per your style it should be ` -- `. All in `01_Introduction.tex`, lines **16, 27, 41, 66, 68, 80, 84, 89, 101**. A scoped find/replace of ` - ` → ` -- ` in that file fixes them in one pass — **but do NOT run it globally**: lines 97 & 114 of `03_Methodology.tex` and line 196 of `04` use ` - ` as a **math minus sign** and must stay.

### 7. Smaller wording / consistency (optional)
- `03_Methodology.tex:327`: "the non-parametric **domain-specific** Wilcoxon" — Wilcoxon isn't domain-specific; drop "domain-specific".
- `appendices/AnAppendix.tex:32`: "Århus" vs "Aarhus" used elsewhere (and in AIAS). Pick one spelling.
- Appendix writes raw `MVP`, `SAM`, `MobileSAM`, `Gaussian` while the main text uses `\acs{...}`. Acronyms are already defined by then, so it's legal, but `\acs{}` would be more consistent.
- `01_Introduction.tex:118`-style leftover: there's a `% TODO: add page ref for the morphological operations chapter` at `03_Methodology.tex:118`. Either add the page ref to `\parencite{DigitalImageProcessing}` or delete the TODO so it can't surface.
- Preface comes before Abstract in `Thesis.tex` (33-34); the DTU template order is Abstract → Preface. Low stakes, but trivial to swap if you want to match.

---

## What's strong (don't second-guess these)
- Supervisor's asks are visibly addressed: the **Consolidated discussion** (§4.5) is the "pure discussion across all RQs"; the **tool architecture + workflow flowcharts** are in; citations are used instead of footnotes; the "from scratch vs. from drawings" pivot and the bigger-dataset / 3D-scan future work are all there.
- Negative result is framed honestly and the limitations (image quality, alignment, n=3, single user, binarization-threshold change) are stated up front rather than buried — examiners reward this.
- The statistical reasoning is otherwise correct (normality checked before the t-test, SEM error bars, exploratory framing for the case studies). Fixing #1 makes this airtight.
- Conclusion follows the slide-21 checklist: restates the aim, answers each RQ, introduces nothing new except future work.

---

## Grade (Danish 7-trinsskala)

These are my honest estimates as a tough examiner; your supervisors set the real mark.

**Report alone: 10 (solidly), with 12 in reach if you fix #1–#4.**
The work is well-scoped, the methodology is thorough and genuinely multi-pronged (classical baselines + 5 SAM variants + ablation + ICL + a real user case study), and a negative result is presented with maturity. What currently holds it under 12: the self-contradicting t-test direction (#1), the caption/image and count inconsistencies (#2, #3), and a handful of typos — these are exactly the "lack of care" signals that move a 12 to a 10. None are deep; all are fixable in your remaining time.

**Report + a strong defense: 12 is realistic.**
A negative-result thesis lives or dies on the defense — you have to show command of *why* SAM failed and what you'd do next. You already have the material (the "from drawings" pivot, epigraphically-tuned SAM, INSCRIBE 3D). If in the 20-min slot you (a) own the limitations before the examiner raises them, (b) explain the t-test/normality choices crisply, and (c) sell the future-work pivot as the real contribution, the negative result reads as rigor rather than failure, and 12 is on the table. A shaky defense that can't justify the design choices would pull it toward 7.

### To push the grade up, in priority order
1. Fix the t-test direction contradiction (#1) — biggest single risk.
2. Fix the PH2/HT7a caption (#2) and the 45/27 count (#3) — cheap credibility wins.
3. Promote the orphan subsection (#4) so the ToC looks clean.
4. Sweep the Introduction dashes (#6) and the minor wording (#7).
5. For the defense: rehearse a 60-second answer to "isn't a negative result just a failed project?" — anchor it on the pivot and the evaluation framework as reusable contributions.

---

## Done — mechanical fixes I already applied
Typos: `contigous`→`contiguous` (×4), `clealy`→`clearly`, `seeting`→`setting`, `strucured`→`structured`, `foregound`→`foreground`, `demonstratively`→`demonstrably` (→ "demonstrably"). Grammar: two sentence-final dangling commas → periods (`03:172`, `04:38`); "did not work. and when" → "…work, and when" (appendix); "a MVP" → "an MVP". Factual: "no need for a **parametric** test like Wilcoxon" → "**non-parametric**" (Wilcoxon is non-parametric; this also now matches `03:327`). Consistency: `\cite{MayaSAM}` → `\parencite{MayaSAM}` (`04:317`).
