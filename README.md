# Meeting Transcriber

Transcribes a meeting audio recording and produces a speaker-labeled SRT file.

**Pipeline:** WhisperX (speech-to-text + word alignment) → pyannote.audio (speaker diarization) → SRT output. Runs fully locally, no per-minute cost after first model download.

## Requirements

**System:**
- Python 3.10+
- ffmpeg (used by WhisperX to decode audio files)

**HuggingFace (free, one-time):**
- A [HuggingFace token](https://huggingface.co/settings/tokens) with **Read** access
- Accept the pyannote model license (while logged in to HuggingFace):
  - https://huggingface.co/pyannote/speaker-diarization-community-1

## Installation

```bash
# 1. Install ffmpeg (required for audio decoding)
# Ubuntu / Debian / WSL:
sudo apt-get install -y ffmpeg
# macOS:
brew install ffmpeg

# 2. Install PyTorch CPU-only (Intel/AMD GPU or no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Install the CLI
pip install -e .
```

## Usage

```bash
# Set your HuggingFace token — either in .env:
#   HF_TOKEN=hf_...
# or as an env var:
export HF_TOKEN=hf_...

# Transcribe a recording
transcribe meeting.mp3

# With options
transcribe meeting.mp3 --speakers 2 --language en --model small --output meeting.srt

# Use the faster standard diarization model (recommended, requires license acceptance)
# Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
#                    https://huggingface.co/pyannote/segmentation-3.0
transcribe meeting.mp3 --speakers 2 --language en

# Use the community diarization model (slower, no extra license needed beyond the token)
transcribe meeting.mp3 --speakers 2 --language en \
  --diarization-model pyannote/speaker-diarization-community-1

# Resume after a failed diarization without re-transcribing
transcribe meeting.mp3 --speakers 2 --language en --resume
```

Output: an SRT file alongside the input audio (or at `--output` path).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--speakers N` | auto | Number of speakers (auto-detected if omitted) |
| `--language CODE` | auto | Language code, e.g. `en`, `ru` (auto-detected if omitted) |
| `--model NAME` | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--diarization-model MODEL` | `pyannote/speaker-diarization-3.1` | Pyannote diarization model |
| `--output PATH` | `<input>.srt` | Output SRT file path |
| `--hf-token TOKEN` | `$HF_TOKEN` | HuggingFace token (overrides env var) |
| `--resume` | off | Reuse cached interim results from `.tmp/` |

## Diarization Models

| Model | Speed (CPU) | Notes |
|-------|-------------|-------|
| `pyannote/speaker-diarization-3.1` | Fast ✓ default | Requires accepting license at hf.co |
| `pyannote/speaker-diarization-community-1` | Very slow | No extra license step needed |

## Speed vs. Accuracy Trade-offs (CPU-only)

| Model | ~Speed (1hr audio) | Accuracy |
|-------|-------------------|----------|
| `tiny` | ~5 min | Low |
| `base` | ~10 min | Fair |
| `small` | ~20 min | Good ✓ default |
| `medium` | ~45 min | Better |
| `large-v3` | ~2–3 hrs | Best |

## SRT Output Format

```
1
00:00:00,000 --> 00:00:05,240
[Speaker 1] Hi, thanks for joining the call.

2
00:00:05,800 --> 00:00:12,100
[Speaker 2] Of course, let's get started.
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

Tests cover timestamp formatting, SRT generation, speaker label mapping, argument validation, HF token resolution, and mocked transcription/diarization pipelines. No models are downloaded when running tests.

## Video Files

Extract audio first with ffmpeg:

```bash
ffmpeg -i recording.mp4 -vn -acodec mp3 recording.mp3
```
