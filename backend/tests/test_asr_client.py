import httpx
import pytest
from asr_client import transcribe_via_asr


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_posts_multipart_and_returns_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json={"full_text": "ok", "diarized": True, "segments": []})

    out = transcribe_via_asr(
        b"audio-bytes", "clip.wav", diarize=True, base_url="http://asr:9000",
        client=_mock_client(handler),
    )
    assert out["full_text"] == "ok"
    assert seen["url"] == "http://asr:9000/transcribe"
    assert b"clip.wav" in seen["body"]
    assert b"true" in seen["body"]  # diarize form field


def test_raises_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(httpx.ConnectError):
        transcribe_via_asr(b"x", "a.wav", diarize=False, base_url="http://asr:9000",
                           client=_mock_client(handler))
