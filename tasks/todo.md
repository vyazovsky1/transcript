# Task List: Meeting Transcriber CLI

## Phase 1: Foundation
- [x] **Task 1** — Project scaffold: `pyproject.toml` + `transcribe.py` CLI stub (`transcribe --help` works)

## Phase 2: Core Pipeline
- [x] **Task 2** — WhisperX transcription + word-level alignment (`run_transcription()`)
- [x] **Task 3** — pyannote diarization + speaker assignment (`run_diarization()`)

### Checkpoint A
- [ ] End-to-end pipeline produces speaker-assigned segments from real audio

## Phase 3: Output + Polish
- [x] **Task 4** — SRT writer: group words by speaker, format valid SRT, map `SPEAKER_00` → `[Speaker 1]`
- [x] **Task 5** — CLI wiring: full arg handling, HF token validation, progress output, error messages

### Checkpoint B
- [ ] Full run on a real meeting file produces a correct, readable `.srt`
- [ ] All acceptance criteria met — ready for real-world testing
