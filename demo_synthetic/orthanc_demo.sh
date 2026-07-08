#!/bin/sh
# Start a live Orthanc (DICOMweb, no auth), then STOW-RS the CT + AI SR into it.
# Requires Docker Desktop running (open -a Docker). Docker-free alternative:
# pacs_ingest_demo.py (pynetdicom C-STORE).
set -e
docker rm -f ich-orthanc 2>/dev/null || true
docker run -d --name ich-orthanc -p 8042:8042 \
    -e ORTHANC__DICOM_WEB__ENABLE=true \
    -e ORTHANC__AUTHENTICATION_ENABLED=false \
    orthancteam/orthanc
echo "waiting for Orthanc ..."
until curl -sf http://localhost:8042/system >/dev/null 2>&1; do sleep 1; done
"$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/orthanc_stow.py"
echo "Orthanc UI: http://localhost:8042/ui/app/  (stop: docker rm -f ich-orthanc)"
