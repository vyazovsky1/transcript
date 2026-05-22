# Meeting Transcriber

## Problem Statement
How might we automatically convert a meeting audio recording into a time-coded, speaker-labeled transcript — locally, for free, with high accuracy?

## Recommended Direction
A minimal CLI tool built on WhisperX + pyannote.audio. WhisperX handles speech-to-text with word-level timestamps; pyannote handles speaker diarization. The two outputs are merged into a single SRT file where each cue is labeled with the speaker identity.

The tool is a single command: `transcribe input.mp3 --speakers 2 --output meeting.srt`. No cloud dependencies, no per-minute cost after initial model download. The only one-time requirement is a free HuggingFace token to download the pyannote model.

If local diarization quality proves insufficient on noisy recordings, the transcription step can be swapped to the OpenAI Whisper API (~$0.006/min) while keeping local diarization — a cheap, targeted upgrade without rewriting the tool.

## Key Assumptions to Validate
- [ ] WhisperX + pyannote produces acceptable speaker separation on 2-person calls — test with a real recording before adding complexity
- [ ] pyannote model download via HuggingFace token is a one-time acceptable setup step
- [ ] Word-level timestamp alignment is accurate enough for clean SRT cue boundaries

## MVP Scope
- Input: local audio file (MP3, WAV, M4A)
- Pipeline: WhisperX transcription → pyannote diarization → timestamp merge
- Output: SRT file with `[Speaker 1]`, `[Speaker 2]` labels per cue
- CLI: `transcribe <file> [--speakers N] [--language en] [--output <path>]`
- Model: whisper large-v3 (best free accuracy); configurable

## Not Doing (and Why)
- Speaker name enrollment — adds setup friction; identity clusters are enough for personal use
- Real-time / live transcription — recordings are the use case; streaming is a different product
- GUI or web interface — CLI is sufficient for personal use
- Video file input — audio extraction is a separate concern; use `ffmpeg -i video.mp4 audio.mp3` as a pre-step
- Cloud SaaS packaging — this is a personal utility, not a product

## Open Questions
- Does pyannote handle overlapping speech gracefully, or does it drop one speaker?
- What's the minimum GPU/CPU requirement for whisper large-v3 to be usable on typical hardware?
