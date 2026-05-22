# Implementation Plan: Meeting Transcriber CLI

## Overview
A single-file Python CLI (`transcribe.py`) that takes a local audio file and produces a speaker-labeled SRT transcript. The pipeline is: WhisperX transcription → word-level alignment → pyannote diarization → speaker assignment → SRT output. Installable via `pip install -e .` so `transcribe` is available as a shell command.

## Architecture Decisions

- **WhisperX over raw Whisper** — WhisperX provides word-level timestamps and has a built-in `assign_word_speakers()` that handles the merge step, removing the need for custom timestamp-overlap logic.
- **Single file (`transcribe.py`)** — no package structure; the whole tool fits in one file with clear function boundaries. Simpler to read, share, and modify.
- **`pyproject.toml` as the install manifest** — lets `pip install -e .` create the `transcribe` CLI entry point without a `setup.py`.
- **HF token via env var `HF_TOKEN` or `--hf-token` flag** — env var is the default for scripts; flag is an override for one-off runs.
- **Model size is configurable, default `small`** — user has Intel Iris Xe (no CUDA, CPU-only). `large-v3` on CPU takes hours for a long meeting; `small` is the practical default. `medium` or `large-v3` are available via `--model` for overnight runs.

## Dependency Graph

```
pyproject.toml + requirements.txt
        │
        ├── transcribe.py: load_audio()
        │           │
        │           ├── run_transcription()   [WhisperX transcribe + align]
        │           │           │
        │           └── run_diarization()     [pyannote via WhisperX]
        │                       │
        │           ┌───────────┘
        │           ▼
        │       assign_speakers()             [whisperx.assign_word_speakers]
        │           │
        │           ▼
        │       write_srt()                   [format → .srt file]
        │           │
        └── CLI entrypoint (argparse)         [wires everything, handles errors]
```

Implementation order is bottom-up: scaffold → transcription → diarization → merge+SRT → CLI polish.

---

## Phase 1: Foundation

### Task 1: Project scaffold — installable package with CLI stub
**Description:** Create `pyproject.toml` and `transcribe.py` so that after `pip install -e .` the command `transcribe --help` works. No real logic yet — just the wiring that proves the entry point is correct.

**Acceptance criteria:**
- [ ] `pyproject.toml` declares all dependencies (whisperx, pyannote-audio, torch) and the `[project.scripts]` entry point `transcribe = "transcribe:main"`
- [ ] `transcribe.py` has a `main()` function with argparse that accepts `audio_file`, `--speakers`, `--language`, `--model`, `--output`, `--hf-token` arguments
- [ ] `transcribe --help` prints usage without error after `pip install -e .`

**Verification:**
- [ ] `pip install -e .` exits 0
- [ ] `transcribe --help` prints argument list

**Dependencies:** None

**Files:**
- `pyproject.toml`
- `transcribe.py`

**Scope:** S

---

## Phase 2: Core Pipeline (vertical slices)

### Task 2: WhisperX transcription + word-level alignment
**Description:** Implement `run_transcription(audio_path, model_name, language, device)` that loads a WhisperX model, transcribes the audio, and runs the alignment step to get word-level timestamps. Returns the aligned result dict (segments with per-word `start`/`end`/`word` fields).

**Acceptance criteria:**
- [ ] Function loads whisperx model and transcribes the given audio file
- [ ] Alignment step runs and result segments contain `words` with `start`, `end`, `word` keys
- [ ] Auto-detects language if `--language` is not provided
- [ ] Works on both CUDA and CPU (device auto-detected)

**Verification:**
- [ ] Run on a short test WAV (≥30s): `transcribe test.wav --model base --output /tmp/test.srt`
- [ ] Inspect raw result: segments have word-level timestamps

**Dependencies:** Task 1

**Files:**
- `transcribe.py` (add `run_transcription()`)

**Scope:** S

### Task 3: pyannote diarization + speaker assignment
**Description:** Implement `run_diarization(audio_path, hf_token, num_speakers, device)` using `whisperx.DiarizationPipeline`. Then call `whisperx.assign_word_speakers(diarize_segments, transcript_result)` to attach a `speaker` field to each word and segment. Returns the enriched result dict.

**Acceptance criteria:**
- [ ] Loads pyannote pipeline with HF token (from arg or `HF_TOKEN` env var)
- [ ] `--speakers N` passes `min_speakers=N, max_speakers=N` to the pipeline
- [ ] Without `--speakers`, diarization runs in auto mode (pyannote infers count)
- [ ] Each segment in the result has a `speaker` field (e.g. `SPEAKER_00`)

**Verification:**
- [ ] Run on same test audio: segments in output have `speaker` key
- [ ] With `--speakers 2`, only two distinct speaker labels appear

**Dependencies:** Task 2 (needs aligned transcript result as input)

**Files:**
- `transcribe.py` (add `run_diarization()`)

**Scope:** S

### Checkpoint: After Tasks 1–3
- [ ] End-to-end pipeline runs: audio in → speaker-assigned segments out
- [ ] No crashes on a real meeting audio file
- [ ] Review speaker separation quality before continuing

---

## Phase 3: Output + Polish

### Task 4: SRT writer
**Description:** Implement `write_srt(segments, output_path)` that formats the speaker-assigned segments as a valid SRT file. Each SRT cue groups consecutive words from the same speaker into one entry. Cue header is `[Speaker 1]`, `[Speaker 2]`, etc. (map `SPEAKER_00` → `Speaker 1`, `SPEAKER_01` → `Speaker 2`).

**Acceptance criteria:**
- [ ] Output is valid SRT: sequential index, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps, speaker-prefixed text, blank line between cues
- [ ] Consecutive words from the same speaker are grouped into one cue (not one cue per word)
- [ ] Speaker labels are human-readable: `[Speaker 1]` not `SPEAKER_00`
- [ ] If `--output` is not given, defaults to `<input_filename>.srt` in the same directory

**Verification:**
- [ ] Open output `.srt` in VLC or a text editor — cues render correctly
- [ ] Validate SRT structure: `python -c "import pysrt; pysrt.open('out.srt')"` (or manual check)

**Dependencies:** Task 3

**Files:**
- `transcribe.py` (add `write_srt()`, `format_timestamp()`, `map_speaker_labels()`)

**Scope:** S

### Task 5: CLI wiring, error handling, and progress output
**Description:** Wire `main()` to call the pipeline functions in order, add user-facing progress prints, and add clear error messages for common failure modes (missing file, missing HF token, unsupported format).

**Acceptance criteria:**
- [ ] Full run: `transcribe meeting.mp3 --speakers 2 --output meeting.srt` works end-to-end
- [ ] Missing audio file: exits with a clear message, non-zero exit code
- [ ] Missing HF token (no `--hf-token` and no `HF_TOKEN` env var): exits with instruction to set it, before attempting model download
- [ ] Progress is visible: at minimum, print which step is running (`Transcribing...`, `Diarizing...`, `Writing SRT...`)
- [ ] `--model` flag correctly passes model name to WhisperX

**Verification:**
- [ ] `transcribe nonexistent.mp3` → clean error, exit 1
- [ ] `transcribe meeting.mp3` with no HF token → clear message before any model loading
- [ ] `transcribe meeting.mp3 --speakers 2 --model base --output out.srt` → produces `out.srt`

**Dependencies:** Task 4

**Files:**
- `transcribe.py` (update `main()`)

**Scope:** S

### Checkpoint: Complete
- [ ] Full end-to-end run on a real meeting recording produces a correct, readable SRT
- [ ] All acceptance criteria above are met
- [ ] Spot-check: speaker separation makes sense for a 2-person meeting
- [ ] Ready for human review / real-world testing

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| pyannote model gated behind HF token | High — blocks first run | Document setup clearly; fail early with a helpful message |
| CPU-only (Intel Iris Xe, no CUDA) | High — large-v3 impractical | Default model is `small`; document speed/accuracy trade-offs in help text |
| pyannote diarization confuses speakers on noisy audio | Med — wrong labels | Validate with a clean test file first; note Whisper API fallback in docs |
| WhisperX API changes between versions | Low — install may break | Pin dependency versions in `pyproject.toml` |
| Overlapping speech — pyannote picks one speaker | Low — minor gaps | Acceptable for MVP; document as known limitation |

## Open Questions
- Does the user's machine have a CUDA GPU? (Affects model choice default — `large-v3` on CPU is slow for long files)
- Should the tool print the transcript to stdout as well as writing the SRT, for quick inspection?
