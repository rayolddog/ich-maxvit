# Orchestrator Guidelines — de novo vs. follow-up branching and interval comparison

*Clinical decision guidelines for the orchestrating agent (Claude, via the
Anthropic API — the intended orchestrator for this pipeline). These extend the
base `SYSTEM_PROMPT` in `ich_agent.py`; the demo wires them via the
`demo_synthetic/` tools. Drafted 2026-07-07.*

## Principle

Detection is stateless (this image, blood or not). The **correct next action is
state-dependent** and the state lives in the chart. After a positive ICH
detection, the orchestrator must decide whether the study is a *de novo*
presentation or a *follow-up*, because that decision changes what the report is
for. This is the one step no static pipeline can perform, and it is the reason
the pipeline is agentic.

## Decision flow (invoke AFTER a positive inference, before writing the report)

```
positive ICH detected
  1. emr_get_patient(patient_id)            → meds (esp. anticoagulants), problems
  2. emr_get_order(accession)               → indication text
  3. pacs_query_priors(patient_id, this_study_datetime)
       ├─ [] no prior head CT → DE NOVO branch
       └─ ≥1 prior head CT    → FOLLOW-UP branch
```

Route on the union of evidence, not any single field: an indication mentioning
"follow-up / known / interval change," OR a qualifying prior study, triggers the
follow-up branch. If the indication says follow-up but no prior is retrievable,
say so explicitly (the prior may be at an outside institution) and proceed as
de novo with a note.

### DE NOVO branch
Characterize the new hemorrhage: subtype(s), location, slice range; state PPV at
the indication-appropriate prevalence; flag acuity. Standard report body.

### FOLLOW-UP branch
The question is no longer "is there blood?" but "**what changed?**"
1. Identify the most recent qualifying prior from `pacs_query_priors`.
2. Call `compare_studies(prior_series_dir, current_series_dir, sdh_side)`.
   The tool re-loads both volumes server-side and returns structured deltas —
   **do not transcribe pixel data or measurements by hand; pass directory
   handles and use the returned numbers verbatim.**
3. Build a COMPARATIVE report keyed to the three interval findings that drive
   management of a subdural collection:
   - **hematoma size** (expansion is the emergency; anticoagulation raises the
     risk — surface it if a DOAC/warfarin is on the med list)
   - **midline shift / mass effect** (new or increased shift can occur without
     the collection enlarging)
   - **ipsilateral parenchymal attenuation** (decreasing HU = developing edema
     / early ischemia)
4. Lead the impression with the *change*, not the presence. Example skeleton:
   "Compared to the prior head CT of [date/time], the left convexity subdural
   hematoma is unchanged in size, but there is new [N] mm midline shift and
   developing hypoattenuation of the ipsilateral white matter, indicating
   progressive mass effect and edema. Correlate clinically; the patient is on
   [anticoagulant]."

## Tool contract (demo)

| Tool | Input | Returns | Notes |
|---|---|---|---|
| `emr_get_patient` | patient_id | demographics, problem list, meds | fictional EMR |
| `emr_get_order` | accession | indication text, study datetime | fictional |
| `pacs_query_priors` | patient_id, before_datetime | list of prior studies (may be empty) | de-novo test = empty list |
| `compare_studies` | prior_dir, current_dir, sdh_side | structured deltas + interpretations | re-loads pixels server-side |

## Complementarity rule (carried from the base design)

The agent is robust with **noisy human language** (indications, histories,
dictation) and must never be trusted to copy **verbatim machine data**
(SOPInstanceUIDs, probabilities, measured HU/mm values). The agent passes opaque
handles; tools re-load and return real data. The comparison numbers in the
report come from `compare_studies`, not from the model's transcription.

## Honesty boundaries to state in any live demo

- The EMR and PACS here are **fictional local mocks**. In production the same
  three calls map to **DICOMweb QIDO** (prior lookup) and **FHIR**
  (`ImagingStudy`, `Condition`, `MedicationStatement`) — real integration, not
  shown here.
- On the phantom, `compare_studies` measures exactly by construction. On real
  exams, registered quantitative comparison (co-registration + segmentation of
  both studies) is a **separate pipeline** the tool would call; the agent
  orchestrates and narrates the delta, it does not itself measure growth.

## Demo case (built in `demo_synthetic/`)

Patient DEMO-0001, on apixaban. Baseline (08:30) small left convexity SDH,
no shift, normal white matter. Follow-up (~12 h later, 20:15) SDH unchanged,
4 mm midline shift, ipsilateral white-matter attenuation 28 → 22 HU. The
follow-up study exercises the full branch; a study with no prior exercises the
de-novo branch for contrast.
