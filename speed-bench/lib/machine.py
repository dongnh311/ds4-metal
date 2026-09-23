"""Guards for benchmarks on the shared 64 GB box (docs/V41_64GB_BUILD.md §4)."""
import re
import subprocess

DS4_PATTERN = r"(^|/)ds4(-server|-agent|-bench|-eval)?( |$)"
_SWAP_RE = re.compile(r"used = ([\d.]+)M")


def ds4_running(pgrep=None, run=subprocess.run):
    """pgrep -fl lines for running ds4 binaries; '' when the machine is free."""
    if pgrep is None:
        proc = run(["pgrep", "-fl", DS4_PATTERN], capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"pgrep failed (rc={proc.returncode}): {proc.stderr.strip()}")
        out = proc.stdout
    else:
        out = pgrep()
    return out.strip()


def parse_swap_used_mib(text):
    m = _SWAP_RE.search(text)
    if not m:
        raise ValueError("unexpected vm.swapusage output")
    return float(m.group(1))


def swap_used_mib():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                         text=True, check=True).stdout
    return parse_swap_used_mib(out)
