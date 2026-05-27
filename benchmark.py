#!/usr/bin/env python3
"""
Benchmark whisper compute_type and thread settings.

Usage:
    python benchmark.py meeting.mp3
    python benchmark.py meeting.mp3 --start 300 --duration 120 --model small
    python benchmark.py meeting.mp3 --no-clip   # use full file
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


CONFIGS = [
    # (compute_type, threads, label)
    ("int8",          0,  "int8         + auto-threads  [baseline]"),
    ("int8",          5,  "int8         + 5 threads"),
    ("int8",          12, "int8         + 12 threads"),
    ("int8_float32",  0,  "int8_float32 + auto-threads"),
    ("int8_float32",  5,  "int8_float32 + 5"),
#    ("int8_float32",  12, "int8_float32 + 12 threads"),
    ("int16",         0,  "int16        + auto-threads"),
#    ("int16",         12, "int16        + 12 threads"),
    ("float32",       0,  "float32      + auto-threads"),
#slow    ("float32",       12, "float32      + 12 threads"),
    # Requires AVX512_BF16 (not available on 13th-gen U-series Intel):
#FAILED    ("int8_bfloat16", 12, "int8_bfloat16 + 12 threads"),
#FAILED    ("bfloat16",      12, "bfloat16     + 12 threads"),
]

DIARIZATION_MODEL_DEFAULT = "pyannote/speaker-diarization-3.1"

# Passed to whisperx.load_model to fill fields added in faster-whisper >= 1.1.0
# that whisperx 3.2.0 doesn't include in its default_asr_options.
_ASR_COMPAT = {"multilingual": False, "hotwords": None}


class _TimedVAD:
    """Wraps a VAD model to record its wall time separately from Whisper inference."""

    def __init__(self, inner):
        self._inner = inner
        self.elapsed: float = 0.0

    def __call__(self, *args, **kwargs):
        t0 = time.monotonic()
        result = self._inner(*args, **kwargs)
        self.elapsed = time.monotonic() - t0
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def clip_audio(input_path: str, duration: int, out_path: str, start: int = 0) -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found — install it or use --no-clip.", file=sys.stderr)
        sys.exit(1)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", input_path,
         "-t", str(duration), "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
        check=True, capture_output=True,
    )


def run_config(
    audio_path: str,
    model_name: str,
    compute_type: str,
    threads: int,
    language: str | None = None,
    align_model=None,
    align_metadata=None,
    diarize_pipeline=None,
    num_speakers: int | None = None,
) -> tuple[float, float, float, float, float | None, str]:
    """Returns (load_s, vad_s, transcribe_s, align_s, diarize_s, text)."""
    import whisperx

    # --- Load whisper model ---
    t0 = time.monotonic()
    model = whisperx.load_model(model_name, "cpu", compute_type=compute_type,
                                threads=threads, language=language,
                                asr_options=_ASR_COMPAT)
    load_time = time.monotonic() - t0

    # --- Transcribe (VAD + Whisper inference) ---
    timed_vad = _TimedVAD(model.vad_model)
    model.vad_model = timed_vad

    t0 = time.monotonic()
    result = model.transcribe(audio_path, language=language)
    total_transcribe = time.monotonic() - t0

    vad_time = timed_vad.elapsed
    transcribe_time = total_transcribe - vad_time

    lang = result.get("language", language or "en")
    del model
    gc.collect()

    # --- Align ---
    if align_model is None:
        align_model, align_metadata = whisperx.load_align_model(language_code=lang, device="cpu")
    t0 = time.monotonic()
    result = whisperx.align(result["segments"], align_model, align_metadata, audio_path, "cpu")
    align_time = time.monotonic() - t0

    # --- Diarize ---
    diarize_time: float | None = None
    if diarize_pipeline is not None:
        kwargs = {"min_speakers": num_speakers, "max_speakers": num_speakers} if num_speakers is not None else {}
        t0 = time.monotonic()
        diarize_segments = diarize_pipeline(audio_path, **kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarize_time = time.monotonic() - t0

    text = " ".join(s["text"].strip() for s in result.get("segments", []))

    return load_time, vad_time, transcribe_time, align_time, diarize_time, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark whisper compute settings.")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model", default="small", metavar="NAME",
                        help="Whisper model (default: small)")
    parser.add_argument("--duration", type=int, default=60, metavar="SEC",
                        help="Clip length in seconds (default: 60)")
    parser.add_argument("--start", type=int, default=0, metavar="SEC",
                        help="Start offset in seconds (default: 0)")
    parser.add_argument("--no-clip", action="store_true",
                        help="Use the full audio file instead of a clip")
    parser.add_argument("--language", default=None, metavar="CODE",
                        help="Language code, e.g. 'en', 'ru' (auto-detected if omitted)")
    parser.add_argument("--speakers", type=int, default=None, metavar="N",
                        help="Expected number of speakers (passed to diarization)")
    parser.add_argument("--hf-token", default=None, metavar="TOKEN",
                        help="HuggingFace token for diarization (or set HF_TOKEN env var)")
    parser.add_argument("--diarization-model", default=DIARIZATION_MODEL_DEFAULT, metavar="MODEL",
                        help=f"Pyannote diarization model (default: {DIARIZATION_MODEL_DEFAULT})")
    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"ERROR: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.no_clip:
            clip_path = args.audio_file
            print(f"Using full file: {args.audio_file}")
        else:
            clip_path = str(Path(tmpdir) / "bench_clip.wav")
            start_fmt = f"{args.start // 60}m{args.start % 60}s" if args.start else "0s"
            print(f"Extracting {args.duration}s clip from {args.audio_file} (start: {start_fmt}) ...")
            clip_audio(args.audio_file, args.duration, clip_path, start=args.start)
            print("Clip ready.\n")

        # Load shared models once (they don't vary across configs)
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        align_model, align_metadata = None, None
        if args.language:
            print(f"Loading alignment model for '{args.language}'...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=args.language, device="cpu")

        diarize_pipeline = None
        if hf_token:
            print(f"Loading diarization model '{args.diarization_model}'...")
            diarize_pipeline = DiarizationPipeline(
                token=hf_token, device="cpu", model_name=args.diarization_model)
        else:
            print("No HF token — diarization phase will be skipped.")

        print()

        results: list[tuple[str, float | None, float | None, float | None, float | None, float | None, str]] = []

        for compute_type, threads, label in CONFIGS:
            print(f"Running: {label} ...", end=" ", flush=True)
            try:
                load_t, vad_t, transcribe_t, align_t, diarize_t, text = run_config(
                    clip_path, args.model, compute_type, threads,
                    language=args.language,
                    align_model=align_model, align_metadata=align_metadata,
                    diarize_pipeline=diarize_pipeline,
                    num_speakers=args.speakers,
                )
                results.append((label, load_t, vad_t, transcribe_t, align_t, diarize_t, text))
                diarize_str = f"  diarize={diarize_t:.1f}s" if diarize_t is not None else ""
                print(f"load={load_t:.1f}s  vad={vad_t:.1f}s  transcribe={transcribe_t:.1f}s  align={align_t:.1f}s{diarize_str}")
            except Exception as exc:
                results.append((label, None, None, None, None, None, ""))
                print(f"FAILED — {exc}")

    # --- timing summary ---
    has_diarize = any(d is not None for _, _, _, _, _, d, _ in results)
    width = 112 if has_diarize else 100
    print("\n" + "=" * width)
    hdr = f"{'Config':<42} {'Load':>6}  {'VAD':>6}  {'Transcr':>7}  {'Align':>7}"
    if has_diarize:
        hdr += f"  {'Diariz':>7}"
    hdr += f"  {'Total':>7}  {'Speedup':>8}"
    print(hdr)
    print("-" * width)

    def _total(load_t, vad_t, transcribe_t, align_t, diarize_t):
        return load_t + vad_t + transcribe_t + align_t + (diarize_t or 0.0)

    baseline_total = next(
        (_total(*ts) for _, *ts, _ in results if ts[0] is not None), None
    )
    for label, load_t, vad_t, transcribe_t, align_t, diarize_t, _ in results:
        if transcribe_t is None:
            print(f"  {label:<40}  {'FAILED':>6}")
        else:
            total_t = _total(load_t, vad_t, transcribe_t, align_t, diarize_t)
            speedup = f"{baseline_total / total_t:.2f}x" if baseline_total else "—"
            flag = " <-- baseline" if total_t == baseline_total else ""
            row = (f"  {label:<40}  {load_t:>5.1f}s  {vad_t:>5.1f}s"
                   f"  {transcribe_t:>6.1f}s  {align_t:>6.1f}s")
            if has_diarize:
                diariz_str = f"{diarize_t:>6.1f}s" if diarize_t is not None else f"{'—':>7}"
                row += f"  {diariz_str}"
            row += f"  {total_t:>6.1f}s  {speedup:>8}{flag}"
            print(row)
    print("=" * width)

    clip_desc = "full file" if args.no_clip else f"{args.duration}s from {args.start}s"
    lang_desc = args.language or "auto"
    speakers_desc = str(args.speakers) if args.speakers else "auto"
    print(f"Model: {args.model}  |  Clip: {clip_desc}  |  Language: {lang_desc}  |  Speakers: {speakers_desc}")
    print("Note: Speedup is based on total wall time. VAD/align/diarize use fixed models and should be roughly constant across configs.")

    # --- save transcripts as files ---
    print("\nTranscripts:")
    stem = Path(args.audio_file).stem
    for label, _, _, transcribe_t, _, _, text in results:
        if transcribe_t is None:
            continue
        safe_label = label.replace(" ", "_").replace("/", "-").replace("[", "").replace("]", "")
        out_path = Path(f"benchmark_{stem}_{safe_label}.txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"  {out_path}")


if __name__ == "__main__":
    main()
