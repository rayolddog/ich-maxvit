"""
Claude API wrapper that logs per-call token usage to CSV.

Designed for the CT head ICH workflow: wrap each Claude call so you can
estimate tokens per study (input + output + cache) and export for analysis.

Usage:
    from claude_token_logger import LoggedClient

    client = LoggedClient(log_path="ich_token_log.csv")

    # Tag each call with a study_id so you can aggregate per-study later
    msg = client.messages_create(
        study_id="CT_0001",
        stage="findings_explanation",   # free-form label for the call's role
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": "..."}],
    )

    # Aggregate after a batch of studies
    client.summarize_per_study()
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


CSV_FIELDS = [
    "timestamp_utc",
    "study_id",
    "stage",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "total_tokens",
    "latency_ms",
    "stop_reason",
]


@dataclass
class CallRecord:
    timestamp_utc: str
    study_id: str
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_tokens: int
    latency_ms: int
    stop_reason: str

    def as_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in CSV_FIELDS}


class LoggedClient:
    def __init__(
        self,
        log_path: str | os.PathLike = "claude_token_log.csv",
        api_key: str | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with self.log_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _append(self, record: CallRecord) -> None:
        with self.log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(record.as_row())

    def messages_create(
        self,
        *,
        study_id: str,
        stage: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        start = time.perf_counter()
        msg = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = msg.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        record = CallRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            study_id=study_id,
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            total_tokens=input_tokens + output_tokens + cache_creation + cache_read,
            latency_ms=latency_ms,
            stop_reason=getattr(msg, "stop_reason", "") or "",
        )
        self._append(record)
        return msg

    def count_tokens(
        self,
        *,
        study_id: str,
        stage: str,
        model: str,
        messages: list[dict[str, Any]],
        log: bool = True,
        **kwargs: Any,
    ) -> int:
        resp = self.client.messages.count_tokens(
            model=model, messages=messages, **kwargs
        )
        n = resp.input_tokens
        if log:
            record = CallRecord(
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                study_id=study_id,
                stage=f"{stage} [count_only]",
                model=model,
                input_tokens=n,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                total_tokens=n,
                latency_ms=0,
                stop_reason="count_only",
            )
            self._append(record)
        return n

    def summarize_per_study(self) -> dict[str, dict[str, int]]:
        """Aggregate the CSV into per-study totals. Returns {study_id: {...}}."""
        totals: dict[str, dict[str, int]] = {}
        if not self.log_path.exists():
            return totals
        with self.log_path.open() as f:
            for row in csv.DictReader(f):
                sid = row["study_id"]
                bucket = totals.setdefault(
                    sid,
                    {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                bucket["calls"] += 1
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "total_tokens",
                ):
                    bucket[k] += int(row[k] or 0)
        return totals

    def print_summary(self) -> None:
        per_study = self.summarize_per_study()
        if not per_study:
            print("No calls logged yet.")
            return

        n = len(per_study)
        sum_in = sum(v["input_tokens"] for v in per_study.values())
        sum_out = sum(v["output_tokens"] for v in per_study.values())
        sum_total = sum(v["total_tokens"] for v in per_study.values())
        print(f"Studies logged: {n}")
        print(f"Avg input tokens/study:  {sum_in / n:,.1f}")
        print(f"Avg output tokens/study: {sum_out / n:,.1f}")
        print(f"Avg total tokens/study:  {sum_total / n:,.1f}")


if __name__ == "__main__":
    # Minimal smoke test — requires ANTHROPIC_API_KEY in env.
    client = LoggedClient(log_path="ich_token_log.csv")
    msg = client.messages_create(
        study_id="CT_DEMO",
        stage="demo",
        model="claude-opus-4-7",
        max_tokens=128,
        messages=[{"role": "user", "content": "Say 'hello' in one word."}],
    )
    print("Response:", msg.content[0].text if msg.content else "(empty)")
    client.print_summary()
