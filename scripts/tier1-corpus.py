#!/usr/bin/env python3
"""Tier-1 parity and resource gate for ffsubsync -> alass --no-split, and the Tier-2 memory gate.

Run through the Dockerfile's CI-only ``corpus`` target, or in any Linux environment with
``alass`` and ``ffsubsync`` (and ``ffmpeg`` for the Tier-2 soundtrack). The fixtures are generated
deterministically; no copyrighted subtitle text or audio is stored or downloaded.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

TIMING = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d)[,.](\d\d\d)")


@dataclass(frozen=True)
class Case:
    name: str
    cues: int
    offset_ms: float = 0
    scale: float = 1
    cut_at: int | None = None
    cut_ms: float = 0
    sparse: bool = False
    overlap: bool = False
    malformed: bool = False
    profile: str = "parallel"


CASES = [
    Case("no-op", 180),
    Case("constant-plus-5s", 240, offset_ms=5_000),
    Case("constant-minus-12s", 240, offset_ms=-12_000),
    Case("fps-25-to-23.976", 600, offset_ms=2_000, scale=25 / 23.976),
    Case("sparse", 12, offset_ms=7_000, sparse=True),
    Case("overlapping-cues", 240, offset_ms=4_000, overlap=True),
    Case("malformed-but-accepted-srt", 180, offset_ms=3_000, malformed=True),
    # These resemble the actual Tier-1 relationship: an English hash-matched anchor and a
    # differently authored foreign-language upload. Cue text, count, boundaries and segmentation
    # do not correspond one-for-one, so identical generated dialogue cannot make the gate pass.
    Case("cross-language-split-merge", 520, offset_ms=4_500, profile="resegmented"),
    Case("missing-and-extra-cues", 700, offset_ms=-3_000, profile="missing-extra"),
    Case("episode", 700, offset_ms=6_000),
    Case("long-film", 2_000, offset_ms=5_000),
    # Neither current Tier-1 implementation is split-aware. This fixture prevents the replacement
    # from being materially worse while documenting why different-cut repair remains Tier 2.
    Case("different-cut-parity", 600, offset_ms=2_000, cut_at=300, cut_ms=30_000),
]

# Tier 2's job: a two-hour film whose subtitle is 5 s late and a further 30 s late after a mid-film
# cut. The soundtrack speaks exactly the reference cues, so the truth is `reference_times`. Cues
# start about 2 s apart, so 3,600 of them fill two hours: denser dialogue than a real film carries,
# which gives the split-aware alignment more to hold, not less.
TIER2_CASE = Case("audio-tier2", 3_600, offset_ms=5_000, cut_at=1_800, cut_ms=30_000)
SOUNDTRACK = Path("/usr/local/share/den-subtitles/soundtrack.mkv")
SAMPLE_RATE = 16_000


def stamp(ms: float) -> str:
    value = max(0, round(ms))
    hours, value = divmod(value, 3_600_000)
    minutes, value = divmod(value, 60_000)
    seconds, millis = divmod(value, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def reference_times(case: Case) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 15_000.0
    for i in range(case.cues):
        gap = (24_000 + (i * 7919) % 19_000) if case.sparse else (900 + (i * 7919) % 2_200)
        cursor += gap
        duration = 850 + (i * 3571) % 2_100
        if case.overlap and i % 7 == 0:
            duration += 2_500
        out.append((cursor, cursor + duration))
    return out


def target_times(case: Case, reference: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for i, (start, end) in enumerate(reference):
        extra = case.cut_ms if case.cut_at is not None and i >= case.cut_at else 0
        out.append((start * case.scale + case.offset_ms + extra, end * case.scale + case.offset_ms + extra))
    return out


def write_srt(
    path: Path,
    times: list[tuple[float, float]],
    malformed: bool = False,
    texts: list[str] | None = None,
) -> list[str]:
    texts = texts or [f"cue {i}: deterministic dialogue {i * 17 % 101}" for i in range(1, len(times) + 1)]
    blocks = [f"{i}\n{stamp(start)} --> {stamp(end)}\n{text}" for i, ((start, end), text) in enumerate(zip(times, texts), 1)]
    body = "\n\n".join(blocks) + "\n"
    if malformed:
        # These are accepted by den's parser: BOM, leading prose, CRLF, padded blank separators,
        # and no final newline. They exercise formatting tolerance without inventing an invalid cue.
        body = body.replace("\n", "\r\n").replace("\r\n\r\n", "\r\n \t\r\n")
        body = "\ufeffexported by editor\r\n\r\n" + body
        body = body.rstrip("\r\n")
    path.write_text(body, encoding="utf-8")
    return texts


def grouped(
    atoms: list[tuple[float, float]],
    sizes: tuple[int, ...],
    language: str,
    drop_every: int | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Render one subtitle author's segmentation of a shared dialogue timeline."""
    times: list[tuple[float, float]] = []
    texts: list[str] = []
    cursor = 0
    group = 0
    while cursor < len(atoms):
        width = sizes[group % len(sizes)]
        chosen = list(range(cursor, min(cursor + width, len(atoms))))
        cursor += width
        group += 1
        chosen = [i for i in chosen if drop_every is None or (i + 1) % drop_every]
        if not chosen:
            continue
        times.append((atoms[chosen[0]][0], atoms[chosen[-1]][1]))
        # Different scripts/languages are intentional. Neither aligner may rely on textual identity.
        if language == "en":
            texts.append(f"We meet after scene {group}; keep the timing natural.")
        else:
            texts.append(f"Nos vemos después de la escena {group}; conserva el ritmo.")
    return times, texts


def fixture(case: Case) -> tuple[list[tuple[float, float]], list[str], list[tuple[float, float]], list[str]]:
    atoms = reference_times(case)
    if case.profile == "parallel":
        texts = [f"cue {i}: deterministic dialogue {i * 17 % 101}" for i in range(1, len(atoms) + 1)]
        return atoms, texts, atoms, texts

    reference, reference_text = grouped(atoms, (1, 2, 1, 3, 1), "en")
    if case.profile == "resegmented":
        target, target_text = grouped(atoms, (2, 1, 3, 2, 1, 1), "es")
        return reference, reference_text, target, target_text

    # Real uploads omit forced/sign cues in one language and add SDH/music cues in another. Give
    # each side different omissions, then add reference-only cues in genuine gaps. This exercises
    # missing/extra evidence without inventing an index correspondence for the oracle.
    reference, reference_text = grouped(atoms, (1, 2, 1, 1), "en", drop_every=19)
    target, target_text = grouped(atoms, (2, 1, 2, 3), "es", drop_every=23)
    extras: list[tuple[tuple[float, float], str]] = []
    for i in range(30, len(atoms), 47):
        previous_end = atoms[i - 1][1]
        next_start = atoms[i][0]
        if next_start - previous_end > 300:
            start = previous_end + 80
            extras.append(((start, min(start + 180, next_start - 20)), f"[music cue {i}]"))
    merged = sorted(zip(reference, reference_text, strict=True), key=lambda row: row[0][0])
    merged.extend(extras)
    merged.sort(key=lambda row: row[0][0])
    return [row[0] for row in merged], [row[1] for row in merged], target, target_text


def parse(path: Path) -> tuple[list[tuple[int, int]], list[str]]:
    body = path.read_text(encoding="utf-8-sig", errors="strict").replace("\r\n", "\n")
    timings = []
    texts = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        match = TIMING.search(line)
        if not match:
            continue
        values = [int(v) for v in match.groups()]
        to_ms = lambda p: ((p[0] * 60 + p[1]) * 60 + p[2]) * 1000 + p[3]
        timings.append((to_ms(values[:4]), to_ms(values[4:])))
        text = lines[i + 1] if i + 1 < len(lines) else ""
        texts.append(text)
    return timings, texts


def canonicalize(path: Path) -> None:
    times, texts = parse(path)
    blocks = [f"{i}\n{stamp(start)} --> {stamp(end)}\n{text}" for i, ((start, end), text) in enumerate(zip(times, texts), 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(command[:2])} failed: {proc.stderr[-500:]}")


def errors(actual: list[tuple[int, int]], expected: list[tuple[float, float]]) -> list[float]:
    if len(actual) != len(expected):
        raise AssertionError(f"cue count changed: {len(expected)} -> {len(actual)}")
    return [abs(a - e) for pair_a, pair_e in zip(actual, expected) for a, e in zip(pair_a, pair_e)]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1)]


def process_tree(pid: int) -> list[tuple[int, str, int]]:
    """(pid, command name, RSS in KB) for `pid` and every descendant."""
    pending, seen, out = [pid], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
            name = re.search(r"^Name:\s+(\S+)", status, re.MULTILINE)
            out.append((current, name.group(1) if name else "?", int(match.group(1)) if match else 0))
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return out


def process_tree_rss_kb(pid: int) -> int:
    return sum(rss for _, _, rss in process_tree(pid))


def timed(command: list[str]) -> tuple[float, int]:
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    peak_rss = 0
    while proc.poll() is None:
        peak_rss = max(peak_rss, process_tree_rss_kb(proc.pid))
        time.sleep(0.005)
    _, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"{' '.join(command[:2])} failed: {stderr[-500:]}")
    return time.perf_counter() - started, peak_rss


def concurrent_peak(commands: list[list[str]]) -> int:
    procs = [subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for command in commands]
    peak_rss = 0
    while any(proc.poll() is None for proc in procs):
        peak_rss = max(peak_rss, sum(process_tree_rss_kb(proc.pid) for proc in procs))
        time.sleep(0.005)
    for command, proc in zip(commands, procs):
        _, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"{' '.join(command[:2])} failed: {stderr[-500:]}")
    return peak_rss


def cgroup_value(name: str) -> int:
    path = Path("/sys/fs/cgroup") / name
    value = path.read_text().strip()
    if value == "max":
        raise RuntimeError(f"{name} is unlimited; run the gate with --memory=512m")
    return int(value)


def cgroup_limit() -> int:
    limit = cgroup_value("memory.max")
    if limit > 512 * 1024 * 1024:
        raise RuntimeError(f"memory.max is {limit}, expected a 512 MiB-or-smaller cgroup")
    return limit


@contextlib.contextmanager
def resident_service(root: Path, alass: str) -> Iterator[None]:
    """Run the real service with the cache/body model forced resident beside it.

    The live cache payload cap is 64 MiB. Python owns and touches 112 MiB here: 80 MiB models that
    cache plus 25% allocator/HashMap overhead, and 32 MiB covers three running Tier-1 jobs plus the
    separately prepared Tier-2 job's capped Strings and network buffers. The real service, shared
    libraries, corpus runner and page cache are charged as well, along with whatever children the
    caller starts inside the block.
    """
    cache_dir = root / "service-cache"
    cache_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PORT": "18093",
            "CACHE_DIR": str(cache_dir),
            "ALASS_PATH": alass,
        }
    )
    service = subprocess.Popen(
        ["den-subtitles"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env
    )
    forced = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if service.poll() is not None:
                _, stderr = service.communicate()
                raise RuntimeError(f"service failed to start: {stderr[-500:]}")
            try:
                with socket.create_connection(("127.0.0.1", 18093), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            raise RuntimeError("service did not listen within 10 seconds")

        forced = bytearray(112 * 1024 * 1024)
        for page in range(0, len(forced), 4096):
            forced[page] = 1
        yield
    finally:
        if service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=3)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()
        forced = None


def full_service_memory_peak(
    root: Path, alass: str, reference: Path, incoming: Path
) -> tuple[int, int, int]:
    """Measure the whole container at the admitted Tier-1 maximum, not just child RSS."""
    limit = cgroup_limit()
    procs: list[subprocess.Popen[str]] = []
    with resident_service(root, alass):
        try:
            commands = [
                [alass, "--no-split", str(reference), str(incoming), str(root / f"full-service-{i}.srt")]
                for i in range(3)
            ]
            procs = [
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                for command in commands
            ]
            peak = cgroup_value("memory.current")
            rss_peak_kb = process_tree_rss_kb(os.getpid())
            while any(proc.poll() is None for proc in procs):
                peak = max(peak, cgroup_value("memory.current"))
                rss_peak_kb = max(rss_peak_kb, process_tree_rss_kb(os.getpid()))
                time.sleep(0.005)
            for command, proc in zip(commands, procs):
                _, stderr = proc.communicate()
                if proc.returncode:
                    raise RuntimeError(f"{' '.join(command[:2])} failed: {stderr[-500:]}")
            peak = max(peak, cgroup_value("memory.current"), cgroup_value("memory.peak"))
            return peak, limit, rss_peak_kb
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


def voice_and_floor(seconds: float) -> tuple[bytes, bytes]:
    """Synthetic dialogue and a room-tone floor, 16-bit mono PCM at SAMPLE_RATE.

    The voice is a 130 Hz harmonic series shaped by two vowel formants, broken into 180 ms syllables
    with 60 ms pauses, so VAD sees many short speech spans the way it does in real dialogue, and the
    split-aware alignment has as many spans to work through. The floor is quiet deterministic noise,
    so the gaps are not digital silence.
    """
    harmonics = []
    for k in range(1, 21):
        f = 130 * k
        weight = math.exp(-(((f - 700) / 300) ** 2)) + 0.6 * math.exp(-(((f - 1_200) / 400) ** 2)) + 0.05 / k
        harmonics.append((2 * math.pi * f, weight))
    norm = sum(weight for _, weight in harmonics)
    voice, floor = array("h"), array("h")
    for n in range(int(seconds * SAMPLE_RATE)):
        noise = (n * 2_654_435_761 >> 13) % 121 - 60
        floor.append(noise)
        t = n / SAMPLE_RATE
        phase = t % 0.24
        if phase >= 0.18:
            voice.append(noise)
            continue
        envelope = math.sin(math.pi * phase / 0.18) ** 2
        sample = sum(weight * math.sin(omega * t) for omega, weight in harmonics) / norm
        voice.append(round(12_000 * envelope * sample) + noise)
    return voice.tobytes(), floor.tobytes()


def tile(pcm: bytes, samples: int) -> bytes:
    size = samples * 2
    return (pcm * (size // len(pcm) + 1))[:size]


def write_soundtrack(path: Path) -> None:
    """Encode TIER2_CASE's dialogue as a two-hour 5.1 E-AC-3 Matroska soundtrack, a film's shape.

    Built into the corpus image, so generating it is never charged to the gate's cgroup.
    """
    voice, floor = voice_and_floor(3)
    times = reference_times(TIER2_CASE)
    if max(end - start for start, end in times) > 3_000:
        raise RuntimeError("a cue outlasts the synthetic voice buffer")
    path.parent.mkdir(parents=True, exist_ok=True)
    encode = [
        "ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
        "-ac", "6", "-ar", "48000", "-c:a", "eac3", "-b:a", "192k", str(path),
    ]
    ffmpeg = subprocess.Popen(encode, stdin=subprocess.PIPE)
    assert ffmpeg.stdin is not None
    cursor = 0
    for start, end in times:
        first, last = round(start * SAMPLE_RATE / 1000), round(end * SAMPLE_RATE / 1000)
        if first > cursor:
            ffmpeg.stdin.write(tile(floor, first - cursor))
            cursor = first
        # Cues can overlap; the voice then carries on from where the previous cue left it.
        if last > cursor:
            ffmpeg.stdin.write(voice[(cursor - first) * 2 : (last - first) * 2])
            cursor = last
    ffmpeg.stdin.write(tile(floor, 60 * SAMPLE_RATE))  # closing credits
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f"ffmpeg exited {ffmpeg.returncode} encoding the soundtrack")
    minutes = (cursor / SAMPLE_RATE + 60) / 60
    print(f"SOUNDTRACK {path} {minutes:.1f} min {path.stat().st_size} bytes")


class MediaRelay(http.server.BaseHTTPRequestHandler):
    """Stands in for `resync::Relay`: serves one file on loopback, with ranges, at `/media`.

    Production streams the film from the network, so its bytes never sit in this cgroup's page
    cache. Each served range is dropped from the cache as it goes, so the gate charges ffmpeg's
    decode and not a copy of the fixture that production never holds.
    """

    media: Path

    def do_HEAD(self) -> None:
        self.respond(body=False)

    def do_GET(self) -> None:
        self.respond(body=True)

    def respond(self, body: bool) -> None:
        size = self.media.stat().st_size
        start, end = 0, size - 1
        requested = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if requested:
            start = int(requested.group(1))
            end = min(int(requested.group(2) or end), end)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", "video/x-matroska")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body:
            return
        fd = os.open(self.media, os.O_RDONLY)
        try:
            offset = start
            while offset <= end:
                chunk = os.pread(fd, min(1 << 20, end - offset + 1), offset)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                os.posix_fadvise(fd, offset, len(chunk), os.POSIX_FADV_DONTNEED)
                offset += len(chunk)
        finally:
            os.close(fd)

    def log_message(self, format: str, *args: object) -> None:
        pass


def cgroup_file_and_anon() -> tuple[int, int]:
    stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
    return int(stat["file"]), int(stat["anon"])


def tier2_memory_peak(root: Path, alass: str, soundtrack: Path) -> dict[str, int]:
    """Measure the container at admission's other maximum: one audio Tier-2 job and one Tier-1 job.

    Run in a fresh container, so `memory.peak` covers only this load. alass is handed a loopback URL
    exactly as `sync_to_audio` hands it the relay, and spawns ffprobe and ffmpeg as its own children,
    which is how they are charged in production. Tier-1 jobs are restarted back to back until Tier 2
    finishes, so one is resident at every point of the audio decode and the alignment after it.
    """
    limit = cgroup_limit()
    long_film = next(case for case in CASES if case.name == "long-film")
    ref_times, ref_texts, target_truth, target_texts = fixture(long_film)
    tier1_ref, tier1_in = root / "tier1-ref.srt", root / "tier1-in.srt"
    write_srt(tier1_ref, ref_times, texts=ref_texts)
    write_srt(tier1_in, target_times(long_film, target_truth), texts=target_texts)
    truth = reference_times(TIER2_CASE)
    tier2_in, tier2_out = root / "tier2-in.srt", root / "tier2-out.srt"
    tier2_texts = write_srt(tier2_in, target_times(TIER2_CASE, truth))
    tier1_command = [alass, "--no-split", str(tier1_ref), str(tier1_in), str(root / "tier1-out.srt")]

    handler = type("Relay", (MediaRelay,), {"media": soundtrack})
    relay = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{relay.server_address[1]}/media"
    own_cgroup = Path("/proc/self/cgroup").read_text()
    stats = dict.fromkeys(
        ["sampled", "file", "anon", "tree_kb", "tier2_alass_kb", "tier1_alass_kb", "ffmpeg_kb", "ffprobe_kb",
         "ffmpeg_seen", "foreign_cgroup", "tier1_runs"],
        0,
    )
    tier2 = tier1 = None
    try:
        with resident_service(root, alass), open(root / "tier2.err", "w+") as tier2_err:
            started = time.perf_counter()
            tier2 = subprocess.Popen(
                [alass, url, str(tier2_in), str(tier2_out)], stdout=subprocess.DEVNULL, stderr=tier2_err
            )
            checked: set[int] = set()
            while tier2.poll() is None:
                if tier1 is None or tier1.poll() is not None:
                    if tier1 is not None:
                        _, stderr = tier1.communicate()
                        if tier1.returncode:
                            raise RuntimeError(f"Tier-1 alass failed: {stderr[-500:]}")
                        stats["tier1_runs"] += 1
                    tier1 = subprocess.Popen(
                        tier1_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
                    )
                current = cgroup_value("memory.current")
                if current > stats["sampled"]:
                    stats["sampled"] = current
                    stats["file"], stats["anon"] = cgroup_file_and_anon()
                tree = process_tree(os.getpid())
                stats["tree_kb"] = max(stats["tree_kb"], sum(rss for _, _, rss in tree))
                for pid, name, rss in tree:
                    if pid == tier2.pid:
                        stats["tier2_alass_kb"] = max(stats["tier2_alass_kb"], rss)
                    elif pid == tier1.pid:
                        stats["tier1_alass_kb"] = max(stats["tier1_alass_kb"], rss)
                    elif name in ("ffmpeg", "ffprobe"):
                        stats[f"{name}_kb"] = max(stats[f"{name}_kb"], rss)
                    if name == "ffmpeg" and pid not in checked:
                        checked.add(pid)
                        stats["ffmpeg_seen"] += 1
                        with contextlib.suppress(FileNotFoundError):
                            if Path(f"/proc/{pid}/cgroup").read_text() != own_cgroup:
                                stats["foreign_cgroup"] += 1
                time.sleep(0.005)
            stats["wall_ms"] = round((time.perf_counter() - started) * 1000)
            stats["memory_peak"] = cgroup_value("memory.peak")
            stats["peak"] = max(stats["sampled"], cgroup_value("memory.current"), stats["memory_peak"])
            stats["limit"] = limit
            if tier2.returncode:
                tier2_err.seek(0)
                raise RuntimeError(f"Tier-2 alass failed: {tier2_err.read()[-500:]}")
    finally:
        for proc in (tier1, tier2):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        relay.shutdown()

    aligned, texts = parse(tier2_out)
    if texts != tier2_texts:
        raise AssertionError("Tier 2 changed cue text/order")
    stats["p95_ms"] = round(percentile(errors(aligned, truth), 0.95))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alass", default="alass")
    parser.add_argument("--ffsubsync", default="ffsubsync")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cgroup-memory-gate", action="store_true")
    parser.add_argument(
        "--tier2-memory-gate",
        action="store_true",
        help="run only the audio Tier-2 + Tier-1 cgroup gate; give it a fresh container",
    )
    parser.add_argument("--soundtrack", type=Path, default=SOUNDTRACK)
    parser.add_argument("--write-soundtrack", type=Path, help="generate the Tier-2 soundtrack and exit")
    parser.add_argument("--memory-gate-mib", type=int, default=400)
    args = parser.parse_args()
    if args.write_soundtrack:
        write_soundtrack(args.write_soundtrack)
        return 0
    for binary in (args.alass,) if args.tier2_memory_gate else (args.alass, args.ffsubsync):
        if not (Path(binary).is_file() or shutil.which(binary)):
            parser.error(f"required binary not found: {binary}")
    gate_bytes = args.memory_gate_mib * 1024 * 1024
    if args.tier2_memory_gate:
        return tier2_gate(args.alass, args.soundtrack, gate_bytes)

    failed = []
    resource: dict[str, list[tuple[float, int]]] = {"ffsubsync": [], "alass": []}
    tier1_peak_kb = 0
    service_peak_bytes = service_limit_bytes = service_rss_peak_kb = 0
    with tempfile.TemporaryDirectory(prefix="den-tier1-corpus-") as raw:
        root = Path(raw)
        for case in CASES:
            ref_times, ref_texts, target_truth, target_texts = fixture(case)
            target = target_times(case, target_truth)
            ref, inc = root / f"{case.name}-ref.srt", root / f"{case.name}-in.srt"
            ffout, alout = root / f"{case.name}-ff.srt", root / f"{case.name}-al.srt"
            write_srt(ref, ref_times, case.malformed, ref_texts)
            write_srt(inc, target, case.malformed, target_texts)
            # Mirror sync_to_reference's process boundary: den accepts tolerant SRT, then emits
            # canonical SRT for alass. This is why malformed exporter output remains in the corpus.
            canonicalize(ref)
            canonicalize(inc)
            ff = [args.ffsubsync, str(ref), "-i", str(inc), "-o", str(ffout)]
            al = [args.alass, "--no-split", str(ref), str(inc), str(alout)]
            try:
                run(ff)
                run(al)
                ff_times, ff_text = parse(ffout)
                al_times, al_text = parse(alout)
                ff_err, al_err = errors(ff_times, target_truth), errors(al_times, target_truth)
                if ff_text != target_texts or al_text != target_texts:
                    raise AssertionError("cue text/order changed")
                ff95, al95 = percentile(ff_err, 0.95), percentile(al_err, 0.95)
                # Allow 100 ms measurement/parser noise over the incumbent. For the deliberately
                # unsupported different-cut case, parity is the gate; ordinary cases must also land
                # within 250 ms at p95.
                ceiling = ff95 + 100 if case.cut_at is not None else max(250, ff95 + 100)
                if al95 > ceiling:
                    raise AssertionError(f"p95 regression: ff={ff95:.0f}ms alass={al95:.0f}ms limit={ceiling:.0f}ms")
                print(f"PASS {case.name:28} p95_ms ff={ff95:8.1f} alass={al95:8.1f}")
                if case.name == "long-film":
                    for _ in range(args.repeats):
                        resource["ffsubsync"].append(timed(ff))
                        resource["alass"].append(timed(al))
                    tier1_commands = []
                    for index in range(3):
                        output = root / f"long-film-al-{index}.srt"
                        tier1_commands.append([args.alass, "--no-split", str(ref), str(inc), str(output)])
                    tier1_peak_kb = concurrent_peak(tier1_commands)
                    if args.cgroup_memory_gate:
                        service_peak_bytes, service_limit_bytes, service_rss_peak_kb = (
                            full_service_memory_peak(root, args.alass, ref, inc)
                        )
            except (AssertionError, RuntimeError, UnicodeError) as error:
                failed.append(f"{case.name}: {error}")
                print(f"FAIL {case.name}: {error}")

    if not failed:
        ff_wall = statistics.median(v[0] for v in resource["ffsubsync"])
        al_wall = statistics.median(v[0] for v in resource["alass"])
        ff_rss = statistics.median(v[1] for v in resource["ffsubsync"])
        al_rss = statistics.median(v[1] for v in resource["alass"])
        print(f"RESOURCE median wall_s ff={ff_wall:.3f} alass={al_wall:.3f} speedup={ff_wall/al_wall:.2f}x")
        print(f"RESOURCE median max_rss_kb ff={ff_rss:.0f} alass={al_rss:.0f} reduction={ff_rss/al_rss:.2f}x")
        print(f"RESOURCE three_tier1_peak_rss_kb={tier1_peak_kb}")
        if ff_wall / al_wall < 2 or ff_rss / al_rss < 2:
            failed.append("resource gate missed: both median runtime and peak RSS must improve by >=2x")
        # Admission's other maximum, one audio Tier-2 job beside one Tier-1 job, is measured by
        # --tier2-memory-gate in a container of its own.
        if args.cgroup_memory_gate:
            headroom = service_limit_bytes - service_peak_bytes
            print(
                "RESOURCE full_service_cgroup_peak_bytes="
                f"{service_peak_bytes} limit_bytes={service_limit_bytes} headroom_bytes={headroom} "
                f"process_tree_peak_rss_kb={service_rss_peak_kb} "
                "forced_cache_and_body_model_bytes=117440512"
            )
            if service_peak_bytes > gate_bytes:
                failed.append(
                    f"full-service memory gate missed: total cgroup peak must stay <= "
                    f"{args.memory_gate_mib} MiB under the 512 MiB production limit"
                )
    return verdict(failed)


def tier2_gate(alass: str, soundtrack: Path, gate_bytes: int) -> int:
    if not soundtrack.is_file():
        raise SystemExit(f"soundtrack not found: {soundtrack} (build it with --write-soundtrack)")
    failed = []
    with tempfile.TemporaryDirectory(prefix="den-tier2-gate-") as raw:
        try:
            stats = tier2_memory_peak(Path(raw), alass, soundtrack)
        except (AssertionError, RuntimeError, UnicodeError) as error:
            print(f"FAIL audio-tier2: {error}")
            return verdict([f"audio-tier2: {error}"])
    print(
        f"RESOURCE tier2_mixed_cgroup_peak_bytes={stats['peak']} limit_bytes={stats['limit']} "
        f"headroom_bytes={stats['limit'] - stats['peak']} memory_peak_bytes={stats['memory_peak']} "
        f"sampled_peak_bytes={stats['sampled']} "
        f"at_sampled_peak_anon_bytes={stats['anon']} at_sampled_peak_file_bytes={stats['file']} "
        "forced_cache_and_body_model_bytes=117440512"
    )
    print(
        f"RESOURCE tier2_mixed_peak_rss_kb process_tree={stats['tree_kb']} tier2_alass={stats['tier2_alass_kb']} "
        f"ffmpeg={stats['ffmpeg_kb']} ffprobe={stats['ffprobe_kb']} tier1_alass={stats['tier1_alass_kb']}"
    )
    print(
        f"RESOURCE tier2_wall_ms={stats['wall_ms']} tier1_runs_alongside={stats['tier1_runs']} "
        f"tier2_p95_ms={stats['p95_ms']}"
    )
    # The measurement is only of Tier 2 if its ffmpeg decode ran, in this cgroup, next to Tier 1.
    if not stats["ffmpeg_seen"] or stats["foreign_cgroup"]:
        failed.append("Tier-2 gate did not observe alass's ffmpeg child inside this cgroup")
    if not stats["tier1_runs"]:
        failed.append("no Tier-1 job completed alongside Tier 2")
    # Tier 2 is split-aware: it must repair both the offset and the mid-film cut against the audio.
    if stats["p95_ms"] > 250:
        failed.append(f"Tier-2 alignment p95 {stats['p95_ms']} ms exceeds 250 ms")
    if stats["peak"] > gate_bytes:
        failed.append(
            f"Tier-2 mixed memory gate missed: total cgroup peak must stay <= "
            f"{gate_bytes // (1024 * 1024)} MiB under the 512 MiB production limit"
        )
    return verdict(failed)


def verdict(failed: list[str]) -> int:
    if failed:
        print("\nGATE FAILED")
        for error in failed:
            print(f"- {error}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
