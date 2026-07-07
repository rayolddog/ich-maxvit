#!/usr/bin/env python3
"""
emr_pacs.py — fictional EMR + PACS for the ICH follow-up demo.

Entirely synthetic. No real patient data, no real database. Provides the three
tools the orchestrating agent calls to make the de-novo-vs-follow-up decision
and to drive the comparison:

    pacs_query_priors(patient_id, before_datetime)  -> list of prior studies
    emr_get_patient(patient_id)                     -> demographics/meds/problems
    emr_get_order(accession)                         -> indication for a study

The "PACS" is the demo_synthetic/pacs/ folder; a study index is (re)built from
the DICOM headers so the query reflects what was actually written. The "EMR" is
a hand-authored fictional record consistent with the two studies.

Build/refresh the index (after generating studies):
    python emr_pacs.py --build
Then the functions above read from the JSON artifacts.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(__file__)
PACS_DIR = os.path.join(HERE, "pacs")
INDEX = os.path.join(HERE, "pacs_index.json")
EMR = os.path.join(HERE, "emr.json")

# Fictional EMR record, consistent with dicom_writer.STUDY_META.
EMR_RECORD = {
    "DEMO-0001": {
        "patient_id": "DEMO-0001",
        "name": "DEMO, Subdural",
        "birth_date": "1951-03-12",
        "sex": "M",
        "problem_list": ["Atrial fibrillation", "Hypertension"],
        "medications": [
            {"name": "apixaban", "class": "DOAC (anticoagulant)",
             "note": "raises hemorrhage risk and expansion risk"},
            {"name": "lisinopril", "class": "ACE inhibitor"},
        ],
        "allergies": ["none known"],
        "encounters": [
            {"date": "2024-01-15", "time": "08:10", "type": "ED",
             "chief_complaint": "Fall at home, struck head; on blood thinner",
             "accession": "DEMOACC001"},
            {"date": "2024-01-15", "time": "20:00", "type": "ED observation",
             "chief_complaint": "Decreased responsiveness, new confusion",
             "accession": "DEMOACC002"},
        ],
    }
}

# indication per accession (what the ordering clinician asked)
ORDERS = {
    "DEMOACC001": {"accession": "DEMOACC001",
                   "indication": "Ground-level fall, head strike; on apixaban. "
                                 "Rule out intracranial hemorrhage.",
                   "study_date": "2024-01-15", "study_time": "08:30"},
    "DEMOACC002": {"accession": "DEMOACC002",
                   "indication": "Known subdural hematoma, ~12 h follow-up; "
                                 "decreased responsiveness. Assess for "
                                 "interval change.",
                   "study_date": "2024-01-15", "study_time": "20:15"},
}


def build_index():
    import pydicom
    studies = {}
    for f in glob.glob(os.path.join(PACS_DIR, "*", "*.dcm")):
        d = pydicom.dcmread(f, stop_before_pixels=True)
        uid = str(d.StudyInstanceUID)
        studies.setdefault(uid, {
            "study_uid": uid,
            "patient_id": str(d.PatientID),
            "accession": str(d.AccessionNumber),
            "study_date": str(d.StudyDate),
            "study_time": str(d.StudyTime),
            "modality": str(d.Modality),
            "description": str(d.StudyDescription),
            "series_dir": os.path.dirname(f),
            "n_instances": 0,
        })
        studies[uid]["n_instances"] += 1
    idx = sorted(studies.values(), key=lambda s: s["study_date"] + s["study_time"])
    json.dump(idx, open(INDEX, "w"), indent=2)
    json.dump(EMR_RECORD, open(EMR, "w"), indent=2)
    print(f"[emr_pacs] indexed {len(idx)} studies -> {INDEX}; wrote {EMR}")
    return idx


# ---- the three agent-facing tools -----------------------------------------
def _dt(date, time):
    return f"{date} {time[:2]}:{time[2:4]}" if len(time) >= 4 else date


def pacs_query_priors(patient_id, before_datetime):
    """Prior studies for a patient strictly before a given 'YYYYMMDD HHMMSS'
    (or 'YYYYMMDD'). Returns [] when none — the de-novo branch."""
    idx = json.load(open(INDEX))
    key = before_datetime.replace("-", "").replace(":", "").replace(" ", "")
    out = []
    for s in idx:
        if s["patient_id"] != patient_id:
            continue
        if (s["study_date"] + s["study_time"]) < key:
            out.append({k: s[k] for k in ("study_uid", "accession",
                        "study_date", "study_time", "modality",
                        "description", "series_dir")})
    return out


def emr_get_patient(patient_id):
    return json.load(open(EMR)).get(patient_id)


def emr_get_order(accession):
    return ORDERS.get(accession)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="print the follow-up branch the agent would take")
    args = ap.parse_args()
    if args.build:
        return build_index()
    if args.demo:
        # simulate what the agent sees when the FOLLOW-UP study arrives
        priors = pacs_query_priors("DEMO-0001", "20240115 201500")
        print(f"prior head CTs before follow-up: {len(priors)}")
        for p in priors:
            print(f"  {p['study_date']} {p['study_time']} {p['description']} "
                  f"acc={p['accession']}")
        pt = emr_get_patient("DEMO-0001")
        meds = ", ".join(m["name"] for m in pt["medications"])
        print(f"patient meds: {meds}")
        print(f"order indication: {emr_get_order('DEMOACC002')['indication']}")


if __name__ == "__main__":
    main()
