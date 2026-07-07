# Synthetic follow-up demo case

Fully fictional. No real patient data, databases, or derived images. Builds the
de-novo-vs-follow-up + interval-comparison demonstration (see
`ORCHESTRATOR_GUIDELINES.md`).

## Build
```
python synthetic_ct.py --selftest     # verify the encoded deltas
python dicom_writer.py                 # write pacs/{baseline,followup}/*.dcm
python emr_pacs.py --build             # index the PACS, write the fictional EMR
python emr_pacs.py --demo              # show the follow-up branch the agent takes
python compare_studies.py              # the interval deltas the report is built on
```

## The case
Patient DEMO-0001, on apixaban (anticoagulant → expansion risk).
- **Baseline** 2024-01-15 08:30 — small left convexity SDH (5 mm, ~4 cm arc),
  no midline shift, normal white matter.
- **Follow-up** 2024-01-15 20:15 (~12 h) — SDH unchanged in size, **4 mm new
  midline shift**, ipsilateral white-matter attenuation **28 → 22 HU**
  (developing edema).

The follow-up exercises the full agent branch; a no-prior study exercises the
de-novo branch for contrast. Each study is 4 axial slices (the pathology-bearing
slices), ~2 MB — real studies carry 200+ images but the comparison reads a
handful.

## Files
- `synthetic_ct.py` — parametric phantom + HU-calibrated measurement helpers
- `dicom_writer.py` — linked CT DICOM series (deterministic UIDs, STUDY_META)
- `emr_pacs.py` — fictional EMR + PACS index + the three agent tools
- `compare_studies.py` — interval-delta tool (re-loads pixels server-side)
- `ORCHESTRATOR_GUIDELINES.md` — the branching + comparison protocol
