"""System-wide wired memory from vm_stat.

RSS and phys_footprint miss Metal residency; the wired page count does not
(docs/V41_64GB_BUILD.md §3.1)."""
import re
import statistics
import subprocess
import threading

GIB = 1024 ** 3
_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_WIRED_RE = re.compile(r"^Pages wired down:\s+(\d+)\.", re.M)


def parse_vm_stat(text):
    """Wired bytes from one vm_stat output."""
    page = _PAGE_RE.search(text)
    pages = _WIRED_RE.search(text)
    if not page or not pages:
        raise ValueError("vm_stat output lacks the page size or wired pages")
    return int(page.group(1)) * int(pages.group(1))


def summarize(samples):
    """Steady state = median of the second half of the samples; peak = max. GiB."""
    if not samples:
        raise ValueError("no wired samples")
    tail = samples[len(samples) // 2:]
    return {"steady_gib": statistics.median(tail) / GIB,
            "peak_gib": max(samples) / GIB,
            "n": len(samples)}


def _read_vm_stat():
    return subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout


class WiredSampler:
    """Samples vm_stat every `interval` seconds while inside a `with` block."""

    def __init__(self, interval=0.5, read=None):
        self.interval = interval
        self.read = read or _read_vm_stat
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(parse_vm_stat(self.read()))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    def summary(self):
        return summarize(self.samples)
