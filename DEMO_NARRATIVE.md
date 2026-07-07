# What This Demo Is Actually About

*The clinical thesis behind the pipeline — the narrative frame for any
demonstration of this repository. Drafted 2026-07-07 from J.B.'s framing;
the code sections below map each claim to the module that implements it.*

## The product is not detection. It is clinician trust calibration.

The consumers of a radiology report — ER physicians, surgeons,
internists/neurologists — are continuously performing a trust calculation:
*can I act on this report?* Three things feed that calculation, and an AI
pipeline integrated into workflow must serve all three:

1. **Accuracy of diagnosis** — dominated by the vision model. Everything else
   in the pipeline inherits its ceiling from the classifier
   (`train_maxvit.py`, calibrated fixed-HU-window preprocessing in
   `hu_windows.py` — the validated 5.5× gray-matter/blood separation).
2. **Rapidity of reporting** — the workflow layer: automated triage of the
   incoming study, inference at arrival, worklist flagging before a human
   opens the exam (`prevalence_scanner.py` stages, `ich_worklist.py`).
3. **Confidence in the report** — the least-served leg in commercial AI, and
   this demo's focus. Three design commitments:
   - **Not canned.** Clinicians discount templated prose (direct
     communication, local ER physicians). The language layer produces a
     *discussion* of findings, not a checklist — this is what the agentic
     mode is for.
   - **Pointed at the images.** Confidence rises sharply when the report
     shows the finding rather than asserting it: standards-native overlays
     (SR SCOORD content items / presentation states) that a workstation can
     toggle, never modifying the diagnostic pixels.
   - **Honestly calibrated, locally.** The report carries prevalence-adjusted
     PPV/NPV — and not from a textbook prior: `prevalence_scanner.py` +
     `prevalence_db.py` measure *this institution's* observed ICH rate so the
     stated PPV is the local one. Incidence varies significantly by locale
     and resources; the trust arithmetic must too.

## Report form should follow PPV

A design principle this project proposes: the richness of an AI result
object should scale with the positive predictive value and decision density
of the context.

- **Screening mammography** (PPV of a callback < 1%, protocolized BI-RADS
  pathway): a minimal, checklist-grade result is appropriate; a discussion-
  grade report per flag would be noise at that prevalence.
- **Suspected ICH** (PPV plausibly 10–26% depending on setting, decisions
  urgent and consequential: OR, reversal agents, transfer): a full
  structured report with narrative discussion, image-anchored findings, and
  explicit posterior probabilities is warranted.

One size of AI output does not fit all exam classes; the SR's depth is a
dial, not a default.

## Why there are two pipelines (agentic AND direct)

`run_demo_direct.py` and `run_demo_agent.py` are a deliberate ablation pair,
not redundancy. The diagnostic path — series selection criteria, inference,
thresholds, measurements — is identical and deterministic in both. The
agentic variant adds exactly one layer: a language model between messy human
input and the structured record. What it contributes is what LLMs are
uniquely robust at: **recovering intent through noise** — typos, grammar
slips, transcription errors, telegraphic clinical histories. The original
motivation for the agentic mode was precisely the transcription-correction
tax on radiologist time.

The complementarity principle (and the fix it mandates): LLMs are robust
exactly where input is noisy human language, and fragile exactly where data
must be copied verbatim (SOPInstanceUIDs, probabilities). Therefore the
agent owns the *prose layer* and passes *opaque handles* for the data layer;
inference results are re-loaded server-side, never transcribed by the model.

## The keystone agentic benefit: state-dependent routing (de novo vs. follow-up)

Detection is *stateless* — this image, blood or not. The clinically correct
next action is *state-dependent*, and the state lives outside the pixels, in
the chart. This is the one benefit no static pipeline can replicate, because
it is a branching clinical-reasoning decision, not language handling or
plumbing:

```
ICH identified
  └─ agent queries chart: prior head CT in a short window?
       ├─ NO  → de novo: characterize, localize, flag acute
       └─ YES → follow-up: the question is no longer "is there blood?"
                but "what CHANGED?" → fetch prior, compare:
                  • hemorrhage size (expansion?)
                  • attenuation (evolving age of blood: hyperacute→chronic)
                  • cerebral edema / midline shift (developing or worsening —
                    the classic short-interval deterioration of subdural)
                → report is a COMPARISON, not a fresh description
```

Two claims follow, and they are the strongest in this document:

1. **This is what an orchestrating agent is actually for.** Not doing the
   steps — deciding *which steps the case calls for*. The de-novo/follow-up
   branch changes what the entire downstream analysis is *for*, based on
   external unstructured state. That is the textbook justification for
   orchestration, and it is the honest answer to "why is this agentic at
   all?"
2. **Comparison-over-time is an under-served gap in imaging AI.** Nearly all
   published ICH models score single exams against a label. But the
   clinically decisive events in ICH are *deltas* — hematoma expansion,
   evolving mass effect, the subdural whose midline shift creeps over hours —
   not presences. An AI that reports "blood present, 0.94" on a follow-up is
   answering a question nobody asked. The agent's role is to ensure the
   *right question* is asked of each exam; on follow-ups the right question is
   almost always the comparison, and current tools skip it. This also
   completes the trust argument: "did they even look at the prior?" is the
   most corrosive doubt in follow-up imaging, and an explicitly comparative
   report answers it directly.

**Honesty boundaries (state these in any demo):**
- *Chart access is real integration, not mockable with a demo folder.*
  Retrieving priors + indication means DICOMweb QIDO against a PACS at
  minimum, ideally FHIR against an EHR (`ImagingStudy`, `Condition`,
  `Encounter`). The demo can show the agent *making the branch* on staged
  priors; live EHR/FHIR retrieval is the productionization step.
- *Registered quantitative comparison is its own pipeline.* "Hematoma grew
  9 mm" requires co-registering and segmenting both studies. Keep the agent's
  claim at "identifies that comparison is required and structures the
  comparative report," not "measures the growth" — until a comparison tool
  exists to call.

## The cautionary precedent this design answers

Mammography CAD is the canonical failure of AI-in-workflow: decades of
approved, reimbursed prompt-marks that large studies ultimately found
valueless — because marks arrived without trust calibration, and readers
rationally learned to ignore them. Every commitment above (no canned prose,
image-anchored findings, honest local PPV) is a specific antidote to a
specific reason CAD failed.

## Demo beats (target sequence)

1. Study arrives → staged triage → inference → worklist flag (rapidity).
2. Open the AI result: a conformant DICOM SR (TID 1500) ingested by a real
   PACS (Orthanc via STOW-RS), with toggleable image overlays (confidence:
   pointed at the images).
3. **The branch:** on a positive case, the agent queries for a prior. Show
   both paths — a de novo read, and a follow-up where the agent pivots to a
   comparative report (size/attenuation/midline-shift deltas) against a
   staged prior. This is the "why agentic" beat.
4. Read two reports side by side: direct-mode checklist vs agent-mode
   discussion, same deterministic findings underneath (the ablation).
5. Show the same case's PPV under textbook prior vs. locally measured
   prevalence from the archive scanner (honest calibration).
6. Close on the Bayesian section: what the clinician should actually believe
   when the flag fires — 74% → 26% and why that is a feature of honesty, not
   a weakness of the model.
