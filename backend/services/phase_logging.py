import time
from typing import Optional


class PhaseTrace:
    def __init__(self, scope: str, subject: Optional[str] = None):
        self.scope = str(scope or "TRACE").strip().upper()
        self.subject = str(subject or "").strip()
        self.total_start = time.perf_counter()

    def start_phase(self, phase_name: str, detail: Optional[str] = None) -> float:
        message = phase_name
        if detail:
            message = f"{message}: {detail}"
        print(f"{self._prefix()} {message}")
        return time.perf_counter()

    def complete_phase(self, phase_name: str, phase_start: float, detail: Optional[str] = None):
        elapsed = time.perf_counter() - phase_start
        total = time.perf_counter() - self.total_start
        suffix = f", {detail}" if detail else ""
        print(f"{self._prefix()} {phase_name} Completed in {elapsed:.2f}s (Total: {total:.2f}s){suffix}")

    def skip_phase(self, phase_name: str, detail: Optional[str] = None):
        total = time.perf_counter() - self.total_start
        suffix = f", {detail}" if detail else ""
        print(f"{self._prefix()} {phase_name} Skipped (Total: {total:.2f}s){suffix}")

    def log(self, message: str, detail: Optional[str] = None):
        suffix = f": {detail}" if detail else ""
        total = time.perf_counter() - self.total_start
        print(f"{self._prefix()} {message}{suffix} (Total: {total:.2f}s)")

    def success(self, detail: Optional[str] = None):
        total = time.perf_counter() - self.total_start
        suffix = f": {detail}" if detail else ""
        print(f"{self._prefix()} SUCCESS{suffix}. Total Time: {total:.2f}s")

    def failure(self, detail: Optional[str] = None):
        total = time.perf_counter() - self.total_start
        suffix = f": {detail}" if detail else ""
        print(f"{self._prefix()} FAILED{suffix}. Total Time: {total:.2f}s")

    def _prefix(self) -> str:
        if self.subject:
            return f"[{self.scope}][{self.subject}]"
        return f"[{self.scope}]"
