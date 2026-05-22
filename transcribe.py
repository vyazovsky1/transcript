import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import whisperx
from dotenv import load_dotenv
from tqdm import tqdm
from whisperx.diarize import DiarizationPipeline

load_dotenv()

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
TMP_DIR = Path.cwd() / ".tmp"

log = logging.getLogger("transcribe")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence noisy third-party loggers
    logging.getLogger("whisperx").setLevel(logging.WARNING)
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)


def _interim_path(audio_path: str, model_name: str, stage: str) -> Path:
    TMP_DIR.mkdir(exist_ok=True)
    stem = Path(audio_path).stem
    return TMP_DIR / f"{stem}_{model_name}_{stage}.json"


def _save_interim(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_interim(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def check_prerequisites(hf_token: str) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    errors: list[str] = []

    if shutil.which("ffmpeg") is None:
        errors.append(
            "ffmpeg not found — required for audio decoding.\n"
            "  Install: sudo apt-get install ffmpeg"
        )

    try:
        HfApi().whoami(token=hf_token)
    except Exception:
        errors.append(
            "HuggingFace token is invalid or network is unavailable.\n"
            "  Get a token: https://huggingface.co/settings/tokens"
        )
        _exit_with_errors(errors)

    try:
        HfApi().model_info(DIARIZATION_MODEL, token=hf_token)
    except GatedRepoError:
        errors.append(
            f"Access denied to diarization model '{DIARIZATION_MODEL}'.\n"
            f"  Accept the license at: https://huggingface.co/{DIARIZATION_MODEL}"
        )
    except RepositoryNotFoundError:
        errors.append(f"Diarization model '{DIARIZATION_MODEL}' not found on HuggingFace.")
    except Exception as e:
        errors.append(f"Could not verify diarization model access: {e}")

    _exit_with_errors(errors)


def _exit_with_errors(errors: list[str]) -> None:
    if errors:
        for msg in errors:
            log.error(msg)
        sys.exit(1)


def load_models(
    model_name: str, language: str | None, hf_token: str, device: str
) -> tuple[Any, Any, Any, Any]:
    log.info("Loading models...")

    log.info("[1/3] Whisper '%s'...", model_name)
    whisper_model = whisperx.load_model(model_name, device, compute_type="int8", language=language)

    if language:
        log.info("[2/3] Alignment model for '%s'...", language)
        align_model, align_metadata = whisperx.load_align_model(language_code=language, device=device)
    else:
        log.info("[2/3] Alignment model — will load after language detection.")
        align_model, align_metadata = None, None

    log.info("[3/3] Diarization model '%s'...", DIARIZATION_MODEL)
    diarize_pipeline = DiarizationPipeline(token=hf_token, device=device)

    log.info("All models ready.")
    return whisper_model, align_model, align_metadata, diarize_pipeline


def load_audio(audio_path: str) -> str:
    path = Path(audio_path)
    if not path.exists():
        log.error("File not found: %s", audio_path)
        sys.exit(1)
    if path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        log.error("Unsupported format '%s'. Use MP3, WAV, M4A, FLAC, or OGG.", path.suffix)
        sys.exit(1)
    return str(path)


def resolve_hf_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        log.error(
            "HuggingFace token required for speaker diarization.\n"
            "  Set it via:  export HF_TOKEN=hf_...\n"
            "  Or pass it:  --hf-token hf_...\n"
            "  Get a token: https://huggingface.co/settings/tokens"
        )
        sys.exit(1)
    return token


def _progress_bar(desc: str) -> tuple[tqdm, list[float]]:
    bar = tqdm(total=100, unit="%", desc=f"  {desc}", bar_format="{l_bar}{bar}| {n:.0f}%")
    return bar, [0.0]


def _update_bar(bar: tqdm, last: list[float], pct: float) -> None:
    delta = pct - last[0]
    if delta > 0:
        bar.update(delta)
        last[0] = pct


def run_transcription(
    audio_path: str,
    model_name: str,
    whisper_model: Any,
    align_model: Any,
    align_metadata: Any,
    language: str | None,
    device: str,
    resume: bool = False,
) -> dict:
    transcribe_cache = _interim_path(audio_path, model_name, "transcribed")
    align_cache = _interim_path(audio_path, model_name, "aligned")

    if resume and (cached := _load_interim(align_cache)) is not None:
        log.info("Aligning — using cached result from %s", align_cache)
        return cached

    if resume and (cached := _load_interim(transcribe_cache)) is not None:
        log.info("Transcribing — using cached result from %s", transcribe_cache)
        result = cached
    else:
        log.info("Transcribing...")
        t0 = time.monotonic()
        bar, last = _progress_bar("Transcribing")
        result = whisper_model.transcribe(
            audio_path, language=language,
            progress_callback=lambda pct: _update_bar(bar, last, pct),
        )
        bar.update(100 - last[0])
        bar.close()
        log.info("Transcribing done in %.1fs", time.monotonic() - t0)
        _save_interim(result, transcribe_cache)
        log.info("Transcription saved to %s", transcribe_cache)

    lang = result.get("language", language or "en")
    if align_model is None:
        log.info("Loading alignment model for detected language '%s'...", lang)
        align_model, align_metadata = whisperx.load_align_model(language_code=lang, device=device)

    log.info("Aligning...")
    t0 = time.monotonic()
    bar, last = _progress_bar("Aligning")
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio_path, device,
        progress_callback=lambda pct: _update_bar(bar, last, pct),
    )
    bar.update(100 - last[0])
    bar.close()
    log.info("Aligning done in %.1fs", time.monotonic() - t0)
    _save_interim(result, align_cache)
    log.info("Alignment saved to %s", align_cache)

    return result


def run_diarization(
    audio_path: str,
    pipeline: Any,
    num_speakers: int | None,
    transcript: dict,
) -> dict:
    log.info("Identifying speakers...")
    kwargs = {}
    if num_speakers is not None:
        kwargs = {"min_speakers": num_speakers, "max_speakers": num_speakers}

    t0 = time.monotonic()
    bar, last = _progress_bar("Diarizing")
    diarize_segments = pipeline(
        audio_path,
        progress_callback=lambda pct: _update_bar(bar, last, pct),
        **kwargs,
    )
    bar.update(100 - last[0])
    bar.close()
    log.info("Diarizing done in %.1fs", time.monotonic() - t0)

    return whisperx.assign_word_speakers(diarize_segments, transcript)


def map_speaker_labels(segments: list[dict]) -> dict[str, str]:
    seen: dict[str, str] = {}
    counter = 1
    for seg in segments:
        raw = seg.get("speaker", "")
        if raw and raw not in seen:
            seen[raw] = f"Speaker {counter}"
            counter += 1
    return seen


def format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], output_path: str) -> None:
    label_map = map_speaker_labels(segments)

    cues = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "").strip()
        raw_speaker = seg.get("speaker", "")

        if start is None or end is None or not text:
            continue

        label = label_map.get(raw_speaker, "Speaker ?")
        cues.append((start, end, label, text))

    merged: list[tuple[float, float, str, str]] = []
    for start, end, label, text in cues:
        if merged and merged[-1][2] == label:
            prev_start, _, prev_label, prev_text = merged[-1]
            merged[-1] = (prev_start, end, prev_label, prev_text + " " + text)
        else:
            merged.append((start, end, label, text))

    lines = []
    for idx, (start, end, label, text) in enumerate(merged, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(f"[{label}] {text}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    log.info("Transcript saved to %s", output_path)


def default_output_path(audio_path: str) -> str:
    return str(Path(audio_path).with_suffix(".srt"))


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe a meeting recording and label speakers.",
    )
    parser.add_argument("audio_file", help="Path to audio file (MP3, WAV, M4A, FLAC, OGG)")
    parser.add_argument(
        "--speakers", type=int, default=None, metavar="N",
        help="Expected number of speakers (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--language", default=None, metavar="CODE",
        help="Language code, e.g. 'en', 'ru' (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model", default="small", metavar="NAME",
        help="Whisper model name: tiny, base, small, medium, large-v3 (default: small)",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Output SRT file path (default: <audio_file>.srt)",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace token for pyannote model (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from cached interim results in .tmp/ (default: start from scratch)",
    )

    args = parser.parse_args()

    audio_path = load_audio(args.audio_file)
    hf_token = resolve_hf_token(args.hf_token)
    output_path = args.output or default_output_path(audio_path)
    device = "cpu"

    log.info("Checking prerequisites...")
    check_prerequisites(hf_token)

    whisper_model, align_model, align_metadata, diarize_pipeline = load_models(
        args.model, args.language, hf_token, device
    )

    transcript = run_transcription(
        audio_path, args.model, whisper_model, align_model, align_metadata, args.language, device,
        resume=args.resume,
    )
    transcript = run_diarization(audio_path, diarize_pipeline, args.speakers, transcript)

    log.info("Writing SRT...")
    write_srt(transcript["segments"], output_path)


if __name__ == "__main__":
    main()
