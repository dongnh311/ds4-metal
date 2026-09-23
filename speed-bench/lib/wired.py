"""System-wide wired memory from vm_stat.

RSS and phys_footprint miss Metal residency; the wired page count does not
(docs/V41_64GB_BUILD.md §3.1)."""
import re
import statistics
import subprocess
import threading
import time

GIB = 1024 ** 3
IDLE_WIRED_LIMIT_GIB = 8.0
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


def window_summary(timed, t_start, t_end, min_samples=3):
    """Steady state from samples inside [t_start, t_end]; falls back to the
    second-half median of all samples when fewer than `min_samples` fall
    inside the window. `timed` is a list of (monotonic time, bytes)."""
    if not timed:
        raise ValueError("no wired samples")
    all_bytes = [b for _, b in timed]
    in_window = [b for t, b in timed if t_start <= t <= t_end]
    if len(in_window) >= min_samples:
        steady = statistics.median(in_window)
        window = "decode"
    else:
        tail = all_bytes[len(all_bytes) // 2:]
        steady = statistics.median(tail)
        window = "fallback"
    return {"steady_gib": steady / GIB, "peak_gib": max(all_bytes) / GIB,
            "window": window, "n": len(all_bytes)}


def _read_vm_stat():
    return subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout


def idle_gib(read=None):
    """Wired GiB from one vm_stat sample, taken before a run starts."""
    read = read or _read_vm_stat
    return parse_vm_stat(read()) / GIB


class WiredSampler:
    """Samples vm_stat every `interval` seconds while inside a `with` block."""

    def __init__(self, interval=0.5, read=None, on_sample=None):
        self.interval = interval
        self.read = read or _read_vm_stat
        self.on_sample = on_sample
        self.samples = []
        self.timed = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            value = parse_vm_stat(self.read())
            self.samples.append(value)
            self.timed.append((time.monotonic(), value))
            if self.on_sample is not None:
                self.on_sample(value)
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
