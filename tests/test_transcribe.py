import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import transcribe as t


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------

def test_format_timestamp_zero():
    assert t.format_timestamp(0.0) == "00:00:00,000"

def test_format_timestamp_minutes():
    assert t.format_timestamp(90.5) == "00:01:30,500"

def test_format_timestamp_hours():
    assert t.format_timestamp(3661.25) == "01:01:01,250"

def test_format_timestamp_millis_rounding():
    assert t.format_timestamp(1.9999) == "00:00:02,000"


# ---------------------------------------------------------------------------
# map_speaker_labels
# ---------------------------------------------------------------------------

def test_map_speaker_labels_ordered():
    segs = [
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
    ]
    mapping = t.map_speaker_labels(segs)
    assert mapping == {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}

def test_map_speaker_labels_missing_speaker_key():
    segs = [{"text": "hello"}, {"speaker": "SPEAKER_00"}]
    mapping = t.map_speaker_labels(segs)
    assert "SPEAKER_00" in mapping
    assert len(mapping) == 1


# ---------------------------------------------------------------------------
# write_srt
# ---------------------------------------------------------------------------

def test_write_srt_basic(tmp_path):
    out = str(tmp_path / "out.srt")
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.0, "text": "World", "speaker": "SPEAKER_01"},
    ]
    t.write_srt(segs, out)
    content = Path(out).read_text()
    assert "[Speaker 1] Hello" in content
    assert "[Speaker 2] World" in content
    assert "00:00:00,000 --> 00:00:02,000" in content

def test_write_srt_merges_consecutive_same_speaker(tmp_path):
    out = str(tmp_path / "out.srt")
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 4.0, "text": "Hi", "speaker": "SPEAKER_01"},
    ]
    t.write_srt(segs, out)
    content = Path(out).read_text()
    # Two cues, not three
    assert content.count("[Speaker 1]") == 1
    assert "Hello there" in content

def test_write_srt_skips_empty_text(tmp_path):
    out = str(tmp_path / "out.srt")
    segs = [
        {"start": 0.0, "end": 1.0, "text": "", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "Hi", "speaker": "SPEAKER_00"},
    ]
    t.write_srt(segs, out)
    content = Path(out).read_text()
    assert content.count("[Speaker") == 1

def test_write_srt_default_output_path():
    assert t.default_output_path("/tmp/meeting.mp3") == "/tmp/meeting.srt"
    assert t.default_output_path("/tmp/call.m4a") == "/tmp/call.srt"


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------

def test_load_audio_file_not_found():
    with pytest.raises(SystemExit):
        t.load_audio("/nonexistent/file.mp3")

def test_load_audio_unsupported_format(tmp_path):
    f = tmp_path / "audio.avi"
    f.touch()
    with pytest.raises(SystemExit):
        t.load_audio(str(f))

def test_load_audio_valid(tmp_path):
    for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        f = tmp_path / f"audio{ext}"
        f.touch()
        assert t.load_audio(str(f)) == str(f)


# ---------------------------------------------------------------------------
# resolve_hf_token
# ---------------------------------------------------------------------------

def test_resolve_hf_token_from_arg(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert t.resolve_hf_token("hf_abc123") == "hf_abc123"

def test_resolve_hf_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert t.resolve_hf_token(None) == "hf_from_env"

def test_resolve_hf_token_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert t.resolve_hf_token("hf_arg") == "hf_arg"

def test_resolve_hf_token_missing_exits(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        t.resolve_hf_token(None)


# ---------------------------------------------------------------------------
# DiarizationPipeline constructor — catches wrong kwarg name
# ---------------------------------------------------------------------------

def test_diarization_pipeline_accepts_token_not_use_auth_token():
    from whisperx.diarize import DiarizationPipeline
    sig = inspect.signature(DiarizationPipeline.__init__)
    params = set(sig.parameters.keys())
    assert "token" in params, "DiarizationPipeline expects 'token', not 'use_auth_token'"
    assert "use_auth_token" not in params, "'use_auth_token' is not a valid parameter"


# ---------------------------------------------------------------------------
# Progress callbacks receive values in 0–100 range
# ---------------------------------------------------------------------------

def test_transcription_progress_callback_range():
    """Simulate what whisperx passes: values already in 0-100."""
    from whisperx import asr
    import inspect as _inspect
    src = _inspect.getsource(asr.FasterWhisperPipeline.transcribe)
    # Confirm the formula is ((idx+1)/total)*100, not a 0-1 fraction
    assert "* 100" in src, "whisperx progress_callback values expected to be 0-100"

def test_diarization_progress_callback_range():
    """Confirm whisperx diarize documents 0-100 range."""
    from whisperx.diarize import DiarizationPipeline
    import inspect as _inspect
    src = _inspect.getsource(DiarizationPipeline.__call__)
    assert "0-100" in src or "0 to 100" in src or "float (0-100)" in src


# ---------------------------------------------------------------------------
# run_transcription — mock whisperx, verify calls and progress bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_models — verify all three models are loaded upfront
# ---------------------------------------------------------------------------

@patch("transcribe.DiarizationPipeline")
@patch("transcribe.whisperx.load_align_model")
@patch("transcribe.whisperx.load_model")
def test_load_models_loads_all_three(mock_load_model, mock_load_align, mock_dp_cls):
    mock_load_model.return_value = MagicMock()
    mock_load_align.return_value = (MagicMock(), {})
    mock_dp_cls.return_value = MagicMock()

    whisper_model, align_model, align_metadata, diarize_pipeline = t.load_models(
        "small", "en", "hf_tok", "cpu"
    )

    mock_load_model.assert_called_once_with("small", "cpu", compute_type="int8", language="en")
    mock_load_align.assert_called_once_with(language_code="en", device="cpu")
    mock_dp_cls.assert_called_once_with(token="hf_tok", device="cpu")
    assert whisper_model is not None
    assert align_model is not None
    assert diarize_pipeline is not None

@patch("transcribe.DiarizationPipeline")
@patch("transcribe.whisperx.load_align_model")
@patch("transcribe.whisperx.load_model")
def test_load_models_skips_align_when_no_language(mock_load_model, mock_load_align, mock_dp_cls):
    mock_load_model.return_value = MagicMock()
    mock_dp_cls.return_value = MagicMock()

    _, align_model, align_metadata, _ = t.load_models("small", None, "hf_tok", "cpu")

    mock_load_align.assert_not_called()
    assert align_model is None
    assert align_metadata is None


# ---------------------------------------------------------------------------
# run_transcription — pre-loaded models passed in
# ---------------------------------------------------------------------------

@patch("transcribe.whisperx.align")
def test_run_transcription_calls_whisperx(mock_align, tmp_path):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "language": "en"}
    mock_align.return_value = {"segments": []}

    with patch("transcribe.TMP_DIR", tmp_path):
        t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu")

    mock_model.transcribe.assert_called_once()
    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs.kwargs.get("language") == "en"
    assert callable(call_kwargs.kwargs.get("progress_callback"))

@patch("transcribe.whisperx.align")
def test_run_transcription_progress_callback_is_0_100(mock_align, tmp_path):
    """Progress bar must not multiply pct by 100 (values are already 0-100)."""
    mock_model = MagicMock()

    def fake_transcribe(path, language=None, progress_callback=None, **kw):
        for v in [25.0, 50.0, 75.0, 100.0]:
            if progress_callback:
                progress_callback(v)
        return {"segments": [], "language": "en"}

    mock_model.transcribe.side_effect = fake_transcribe

    def fake_align(segs, model, meta, audio, device, progress_callback=None, **kw):
        if progress_callback:
            progress_callback(100.0)
        return {"segments": []}

    mock_align.side_effect = fake_align

    with patch("transcribe.TMP_DIR", tmp_path):
        with patch("transcribe.tqdm") as mock_tqdm:
            bar = MagicMock()
            mock_tqdm.return_value = bar
            t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu")

    updates = [c.args[0] for c in bar.update.call_args_list]
    for u in updates:
        assert u <= 100, f"Progress bar updated by {u}, expected ≤100 (values are already 0-100)"

@patch("transcribe.whisperx.align")
def test_run_transcription_saves_interim_files(mock_align, tmp_path):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [{"text": "hi"}], "language": "en"}
    mock_align.return_value = {"segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

    with patch("transcribe.TMP_DIR", tmp_path):
        t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu")

    assert (tmp_path / "audio_small_transcribed.json").exists()
    assert (tmp_path / "audio_small_aligned.json").exists()

@patch("transcribe.whisperx.align")
def test_run_transcription_uses_align_cache_when_resume(mock_align, tmp_path):
    """With --resume: if aligned cache exists, skip both transcription and alignment."""
    cached = {"segments": [{"text": "cached", "start": 0.0, "end": 1.0}]}
    (tmp_path / "audio_small_aligned.json").write_text(json.dumps(cached), encoding="utf-8")

    mock_model = MagicMock()
    with patch("transcribe.TMP_DIR", tmp_path):
        result = t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu", resume=True)

    mock_model.transcribe.assert_not_called()
    mock_align.assert_not_called()
    assert result == cached

@patch("transcribe.whisperx.align")
def test_run_transcription_uses_transcribe_cache_when_resume(mock_align, tmp_path):
    """With --resume: if transcription cache exists but align cache doesn't, skip transcription only."""
    cached_transcription = {"segments": [], "language": "en"}
    (tmp_path / "audio_small_transcribed.json").write_text(json.dumps(cached_transcription), encoding="utf-8")
    mock_align.return_value = {"segments": []}

    mock_model = MagicMock()
    with patch("transcribe.TMP_DIR", tmp_path):
        t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu", resume=True)

    mock_model.transcribe.assert_not_called()
    mock_align.assert_called_once()

@patch("transcribe.whisperx.align")
def test_run_transcription_ignores_cache_without_resume(mock_align, tmp_path):
    """Without --resume: cache is ignored even if it exists."""
    cached = {"segments": [{"text": "cached"}]}
    (tmp_path / "audio_small_aligned.json").write_text(json.dumps(cached), encoding="utf-8")
    (tmp_path / "audio_small_transcribed.json").write_text(json.dumps(cached), encoding="utf-8")
    mock_align.return_value = {"segments": []}

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "language": "en"}

    with patch("transcribe.TMP_DIR", tmp_path):
        t.run_transcription("audio.mp3", "small", mock_model, MagicMock(), {}, "en", "cpu", resume=False)

    mock_model.transcribe.assert_called_once()
    mock_align.assert_called_once()


# ---------------------------------------------------------------------------
# run_diarization — pre-loaded pipeline passed in
# ---------------------------------------------------------------------------

@patch("transcribe.whisperx.assign_word_speakers")
def test_run_diarization_uses_pre_loaded_pipeline(mock_assign):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = MagicMock()
    mock_assign.return_value = {"segments": []}

    t.run_diarization("audio.mp3", mock_pipeline, None, {"segments": []})

    mock_pipeline.assert_called_once()

@patch("transcribe.whisperx.assign_word_speakers")
def test_run_diarization_passes_speaker_count(mock_assign):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = MagicMock()
    mock_assign.return_value = {"segments": []}

    t.run_diarization("audio.mp3", mock_pipeline, 2, {"segments": []})

    call_kwargs = mock_pipeline.call_args.kwargs
    assert call_kwargs.get("min_speakers") == 2
    assert call_kwargs.get("max_speakers") == 2
