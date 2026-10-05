import json
import urllib.request

import pytest

from ur3_vision_planning.llm_planner import (
    DEFAULT_MODEL,
    LLMPlanner,
    LLMPlanningError,
    _chat_completions_url,
    _models_url,
)


def test_appends_openai_chat_endpoint():
    assert (
        _chat_completions_url("http://localhost:20128/v1/")
        == "http://localhost:20128/v1/chat/completions"
    )


def test_preserves_full_endpoint():
    endpoint = "https://example.invalid/v1/chat/completions"
    assert _chat_completions_url(endpoint) == endpoint


def test_builds_models_endpoint_from_chat_endpoint():
    assert (
        _models_url("http://localhost:20128/v1/chat/completions")
        == "http://localhost:20128/v1/models"
    )


def test_selected_default_model_keeps_gemini_prefix():
    assert DEFAULT_MODEL == "gemini/gemini-3.8-flash"


class FakePlanner:
    api_key = "secret-not-logged"
    models_url = "http://localhost:20128/v1/models"
    model = DEFAULT_MODEL

    def _check_configuration(self):
        return None

    def _request_json(self, _request, _operation):
        return {"data": [{"id": DEFAULT_MODEL}, {"id": "another/model"}]}


def test_model_catalog_accepts_exact_selected_model():
    models = LLMPlanner.list_models(FakePlanner())
    assert DEFAULT_MODEL in models


def test_model_catalog_allows_unlisted_dynamic_model():
    planner = FakePlanner()
    planner.model = "gemini/not-present"
    models = LLMPlanner.list_models(planner)
    assert planner.model not in models


def test_empty_dynamic_model_catalog_is_advisory():
    planner = FakePlanner()
    planner._request_json = lambda _request, _operation: {"data": []}
    assert LLMPlanner.list_models(planner) == set()


class FakeChatPlanner:
    api_url = "http://localhost:20128/v1/chat/completions"
    api_key = "secret-not-logged"
    model = DEFAULT_MODEL
    _chat_completion = LLMPlanner._chat_completion
    _message_content = staticmethod(LLMPlanner._message_content)

    def _check_configuration(self):
        return None

    def _request_json(self, request, _operation):
        self.sent_payload = json.loads(request.data.decode("utf-8"))
        return {
            "choices": [{"message": {"content": "CONNECTION OK"}}]
        }


def test_connectivity_check_uses_plain_text_content_and_stream_false():
    planner = FakeChatPlanner()
    assert LLMPlanner.check_connection(planner) == "CONNECTION OK"
    assert planner.sent_payload == {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Reply with only: CONNECTION OK"}
        ],
        "stream": False,
    }


def test_connectivity_check_does_not_accept_other_content():
    planner = FakeChatPlanner()

    def wrong_content(_messages, _operation, json_mode=False):
        assert not json_mode
        return "not JSON and not the expected text"

    planner._chat_completion = wrong_content
    with pytest.raises(LLMPlanningError, match="unexpected content"):
        LLMPlanner.check_connection(planner)


class FakeResponse:
    def __init__(self, body, content_type="application/json", status=200):
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.status = status
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        self.read_count += 1
        return self.body


class FakeTransportPlanner:
    timeout = 1
    _body_preview = staticmethod(LLMPlanner._body_preview)
    _validate_request_headers = staticmethod(LLMPlanner._validate_request_headers)


def fake_http_request():
    return urllib.request.Request(
        "http://localhost:20128/v1/test",
        headers={"Accept": "application/json"},
    )


def test_http_body_is_read_once_and_then_parsed(monkeypatch):
    response = FakeResponse(b'{"choices": []}')
    monkeypatch.setattr(
        "ur3_vision_planning.llm_planner.urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )
    result = LLMPlanner._request_json(
        FakeTransportPlanner(), fake_http_request(), "test request"
    )
    assert result == {"choices": []}
    assert response.read_count == 1


def test_sse_response_is_rejected_with_body_preview(monkeypatch):
    response = FakeResponse(b"data: {}\n\n", "text/event-stream")
    monkeypatch.setattr(
        "ur3_vision_planning.llm_planner.urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(LLMPlanningError, match="returned SSE"):
        LLMPlanner._request_json(
            FakeTransportPlanner(), fake_http_request(), "test request"
        )


def test_invalid_http_json_reports_short_body(monkeypatch):
    response = FakeResponse(b"upstream returned an empty event")
    monkeypatch.setattr(
        "ur3_vision_planning.llm_planner.urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(LLMPlanningError, match="body='upstream returned"):
        LLMPlanner._request_json(
            FakeTransportPlanner(), fake_http_request(), "test request"
        )


def test_non_latin1_authorization_header_reports_name_and_index_only():
    secret = "Bearer valid-prefix-đ-secret"
    request = urllib.request.Request(
        "http://localhost:20128/v1/models",
        headers={"Authorization": secret},
    )
    with pytest.raises(LLMPlanningError) as captured:
        LLMPlanner._validate_request_headers(request, "GET /models")
    message = str(captured.value)
    assert "'Authorization'" in message
    assert f"index {secret.index('đ')}" in message
    assert secret not in message
    assert "đ" not in message


def test_vietnamese_prompt_is_utf8_json_body_not_header():
    planner = FakeChatPlanner()
    content = LLMPlanner._chat_completion(
        planner,
        [{"role": "user", "content": "Đưa khối màu vàng vào vùng A."}],
        "POST plan",
        json_mode=True,
    )
    assert content == "CONNECTION OK"
    assert planner.sent_payload["messages"][0]["content"].startswith("Đưa")
    assert planner.sent_payload["stream"] is False
