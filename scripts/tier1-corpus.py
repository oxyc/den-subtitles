#!/usr/bin/env python3
"""Tier-1 parity and resource gate for ffsubsync -> alass --no-split.

Run through the Dockerfile's CI-only ``corpus`` target, or in any Linux environment with
``alass`` and ``ffsubsync``. The fixtures are generated deterministically; no copyrighted
subtitle text is stored or downloaded.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
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


def process_tree_rss_kb(pid: int) -> int:
    pending, seen, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
            total += int(match.group(1)) if match else 0
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(int(child) for child in children)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total


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


def full_service_memory_peak(
    root: Path, alass: str, reference: Path, incoming: Path
) -> tuple[int, int, int]:
    """Measure the whole container at the admitted Tier-1 maximum, not just child RSS.

    The live cache payload cap is 64 MiB. Python owns and touches 112 MiB here: 80 MiB models that
    cache plus 25% allocator/HashMap overhead, and 32 MiB covers three running Tier-1 jobs plus the
    separately prepared Tier-2 job's capped Strings and network buffers. The real service, shared
    libraries, corpus runner, page cache and three real alass children are charged as well.
    """
    limit = cgroup_value("memory.max")
    if limit > 512 * 1024 * 1024:
        raise RuntimeError(f"memory.max is {limit}, expected a 512 MiB-or-smaller cgroup")

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
    procs: list[subprocess.Popen[str]] = []
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
        if service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=3)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()
        forced = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alass", default="alass")
    parser.add_argument("--ffsubsync", default="ffsubsync")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cgroup-memory-gate", action="store_true")
    args = parser.parse_args()
    for binary in (args.alass, args.ffsubsync):
        if not (Path(binary).is_file() or shutil.which(binary)):
            parser.error(f"required binary not found: {binary}")

    failed = []
    resource: dict[str, list[tuple[float, int]]] = {"ffsubsync": [], "alass": []}
    tier1_peak_kb = mixed_proxy_peak_kb = 0
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
                    # Admission weights one split-aware Tier-2 job as two Tier-1 units and reserves
                    # the third for cheap work. Subtitle-reference split mode is a conservative DP
                    # memory proxy; the production Tier-2 benchmark must still include ffmpeg/audio.
                    mixed_proxy_peak_kb = concurrent_peak(
                        [
                            [args.alass, str(ref), str(inc), str(root / "long-film-split.srt")],
                            [args.alass, "--no-split", str(ref), str(inc), str(root / "long-film-cheap.srt")],
                        ]
                    )
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
        print(f"RESOURCE split_plus_tier1_proxy_peak_rss_kb={mixed_proxy_peak_kb}")
        if ff_wall / al_wall < 2 or ff_rss / al_rss < 2:
            failed.append("resource gate missed: both median runtime and peak RSS must improve by >=2x")
        # The full-service gate below runs three Tier-1 children. Admission's other maximum, one
        # Tier-2 job plus one Tier-1 job, is covered by it only while its proxy peak stays lower.
        if mixed_proxy_peak_kb > tier1_peak_kb:
            failed.append(
                "admission memory gate missed: split-plus-Tier-1 proxy exceeds the gated "
                "three-Tier-1 peak"
            )
        if args.cgroup_memory_gate:
            headroom = service_limit_bytes - service_peak_bytes
            print(
                "RESOURCE full_service_cgroup_peak_bytes="
                f"{service_peak_bytes} limit_bytes={service_limit_bytes} headroom_bytes={headroom} "
                f"process_tree_peak_rss_kb={service_rss_peak_kb} "
                "forced_cache_and_body_model_bytes=117440512"
            )
            if service_peak_bytes > 400 * 1024 * 1024:
                failed.append(
                    "full-service memory gate missed: total cgroup peak must leave >=112 MiB "
                    "under the 512 MiB production limit"
                )

    if failed:
        print("\nGATE FAILED")
        for error in failed:
            print(f"- {error}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
