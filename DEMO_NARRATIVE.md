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
3. Read the two reports side by side: direct-mode checklist vs agent-mode
   discussion, same deterministic findings underneath (the ablation).
4. Show the same case's PPV under textbook prior vs. locally measured
   prevalence from the archive scanner (honest calibration).
5. Close on the Bayesian section: what the clinician should actually believe
   when the flag fires — 74% → 26% and why that is a feature of honesty, not
   a weakness of the model.
