from asr_service.config import load_settings

def test_defaults(monkeypatch):
    monkeypatch.delenv("ASR_MODEL_REPO", raising=False)
    monkeypatch.delenv("ASR_DEVICE", raising=False)
    s = load_settings()
    assert s.model_repo == "mlx-community/whisper-large-v3-turbo"
    assert s.device == "mps"
    assert s.hf_offline is True
    assert s.port == 9000

def test_env_override(monkeypatch):
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_PORT", "9100")
    s = load_settings()
    assert s.device == "cpu"
    assert s.port == 9100
