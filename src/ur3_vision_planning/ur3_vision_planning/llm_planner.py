"""OpenAI-compatible LLM planner. There is deliberately no mock fallback."""

import json
import os
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node


class LLMPlanningError(RuntimeError):
    pass


DEFAULT_9ROUTER_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "gemini/gemini-3.8-flash"


def _configured_model(default=DEFAULT_MODEL):
    return os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", default))


def _api_root(base_url):
    base_url = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def _chat_completions_url(base_url):
    return f"{_api_root(base_url)}/chat/completions"


def _models_url(base_url):
    return f"{_api_root(base_url)}/models"


class LLMPlanner(Node):
    def __init__(self, student_config=None, default_model=DEFAULT_MODEL):
        super().__init__("llm_planner")
        student_config = student_config or {}
        mapping = student_config.get("zone_mapping", {})
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.declare_parameter(
            "openai_base_url",
            os.getenv("OPENAI_BASE_URL", DEFAULT_9ROUTER_URL),
        )
        self.declare_parameter(
            "llm_model",
            _configured_model(default_model),
        )
        self.base_url = self.get_parameter("openai_base_url").value.strip()
        self.model = self.get_parameter("llm_model").value.strip()
        self.api_url = _chat_completions_url(self.base_url)
        self.models_url = _models_url(self.base_url)
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        student_name = student_config.get("student_name", "Lê Hồng Quang")
        student_id = str(student_config.get("student_id", "23020757"))
        if len(student_id) < 2 or not student_id[-2:].isdigit():
            raise ValueError("student_id must end in two digits")
        xx = int(student_id[-2:])
        permutation = xx % 6
        zone_a = mapping.get("zone_a", "yellow_cube")
        zone_b = mapping.get("zone_b", "blue_cube")
        zone_c = mapping.get("zone_c", "red_cube")

        self.system_prompt = f"""You are a robot task planner. Understand Vietnamese and English.
Return exactly one JSON object and no Markdown or explanation.

Schema: {{"plan": [steps]}}
Allowed steps (no other fields):
- {{"skill":"pick","object":"<object>"}}
- {{"skill":"place","object":"<same held object>","zone":"<zone>"}}
- {{"skill":"home"}}
Allowed objects: red_cube, yellow_cube, blue_cube.
Allowed zones: zone_a, zone_b, zone_c.

Rules:
- Never output trajectories, joints, controller commands, or unsupported skills.
- A place must follow a matching pick. Never pick while holding another object.
- End every plan with home.
- Do not invent a temporary zone. The executor will reject occupied targets.

Student: {student_name}; ID: {student_id}; XX={xx}; P={xx} mod 6={permutation}.
Personalized mapping: zone_a={zone_a}, zone_b={zone_b}, zone_c={zone_c}.
For "Sắp xếp tất cả vật theo mã sinh viên của tôi" or its English equivalent,
pick/place all three objects according to that mapping, then home.
"""

    def _check_configuration(self):
        if not self.base_url:
            raise LLMPlanningError("OPENAI_BASE_URL/openai_base_url is not set")
        if not self.api_key:
            raise LLMPlanningError("OPENAI_API_KEY is not set")
        if not self.model:
            raise LLMPlanningError("OPENAI_MODEL/LLM_MODEL/llm_model is not set")

    def _request_json(self, request, operation):
        self._validate_request_headers(request, operation)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.getcode()
                content_type = response.headers.get("Content-Type", "")
                raw_body = response.read()
        except UnicodeEncodeError as error:
            # This may come from an implicit header added inside urllib (for
            # example Host or proxy authorization), after explicit headers
            # have already passed the check above. Never include its value.
            raise LLMPlanningError(
                f"9Router {operation} could not encode an implicit HTTP "
                f"header at index {error.start}; check URL/proxy configuration"
            ) from error
        except urllib.error.HTTPError as error:
            content_type = error.headers.get("Content-Type", "")
            raw_body = error.read()
            detail = self._body_preview(raw_body)
            if error.code == 401:
                raise LLMPlanningError(
                    f"9Router {operation} HTTP 401 ({content_type}): "
                    f"verify OPENAI_API_KEY; body={detail}"
                ) from error
            if error.code == 404:
                if operation.startswith("POST"):
                    raise LLMPlanningError(
                        f"9Router model {self.model!r} is unavailable: {detail}"
                    ) from error
                raise LLMPlanningError(
                    f"9Router {operation} HTTP 404: verify OPENAI_BASE_URL"
                ) from error
            raise LLMPlanningError(
                f"9Router {operation} HTTP {error.code} "
                f"({content_type}): body={detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise LLMPlanningError(
                f"9Router {operation} connection failed: {error}"
            ) from error

        if status < 200 or status >= 300:
            raise LLMPlanningError(
                f"9Router {operation} returned HTTP {status} "
                f"({content_type}): body={self._body_preview(raw_body)}"
            )
        if not raw_body:
            raise LLMPlanningError(
                f"9Router {operation} returned an empty HTTP {status} response "
                f"({content_type or 'no Content-Type'})"
            )
        if "text/event-stream" in content_type.lower():
            raise LLMPlanningError(
                f"9Router {operation} returned SSE although stream=false: "
                f"body={self._body_preview(raw_body)}"
            )
        if "json" not in content_type.lower():
            raise LLMPlanningError(
                f"9Router {operation} returned unexpected Content-Type "
                f"{content_type!r}: body={self._body_preview(raw_body)}"
            )
        try:
            body_text = raw_body.decode("utf-8")
            return json.loads(body_text)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise LLMPlanningError(
                f"9Router {operation} HTTP {status} ({content_type}) response "
                f"is not valid JSON: {error}; body={self._body_preview(raw_body)}"
            ) from error

    @staticmethod
    def _validate_request_headers(request, operation):
        """Validate every explicit header without logging any header value."""
        for header_name, header_value in request.header_items():
            try:
                header_name.encode("ascii")
            except UnicodeEncodeError as error:
                raise LLMPlanningError(
                    f"9Router {operation} header name cannot be encoded as "
                    f"ASCII at index {error.start}; header name is not logged"
                ) from error
            try:
                str(header_value).encode("latin-1")
            except UnicodeEncodeError as error:
                source = (
                    "OPENAI_API_KEY environment variable used by this node"
                    if header_name.lower() == "authorization"
                    else "request configuration"
                )
                raise LLMPlanningError(
                    f"9Router {operation} header {header_name!r} cannot be "
                    f"encoded as latin-1 at index {error.start}; check {source}"
                ) from error

    @staticmethod
    def _body_preview(raw_body, limit=500):
        if not raw_body:
            return "<empty>"
        text = raw_body.decode("utf-8", errors="replace")
        return repr(text[:limit])

    @staticmethod
    def _message_content(result, operation):
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMPlanningError(
                f"9Router {operation} response has no "
                "choices[0].message.content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise LLMPlanningError(
                f"9Router {operation} returned empty message content"
            )
        return content.strip()

    def _chat_completion(self, messages, operation, json_mode=False):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload.update(
                {
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                }
            )
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        result = self._request_json(request, operation)
        return self._message_content(result, operation)

    def check_connection(self):
        """Match the known-good curl request; content here is plain text."""
        self._check_configuration()
        content = self._chat_completion(
            [{"role": "user", "content": "Reply with only: CONNECTION OK"}],
            "POST /chat/completions connectivity check",
            json_mode=False,
        )
        if content != "CONNECTION OK":
            raise LLMPlanningError(
                f"9Router connectivity check returned unexpected content: {content!r}"
            )
        return content

    def list_models(self):
        """Return the advisory model catalog; dynamic models may be absent."""
        self._check_configuration()
        request = urllib.request.Request(
            self.models_url,
            method="GET",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        result = self._request_json(request, "GET /models")
        try:
            model_ids = {
                item["id"] for item in result["data"]
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
        except (KeyError, TypeError) as error:
            raise LLMPlanningError(
                "9Router GET /models response has no valid data array"
            ) from error
        return model_ids

    def plan(self, command):
        if not isinstance(command, str) or not command.strip():
            raise LLMPlanningError("User command is empty")
        self._check_configuration()

        return self._chat_completion(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command.strip()},
            ],
            "POST /chat/completions plan",
            json_mode=True,
        )


def main(args=None):
    rclpy.init(args=args)
    planner = LLMPlanner()
    try:
        try:
            model_ids = planner.list_models()
            if planner.model not in model_ids:
                print(
                    f"9ROUTER WARNING: configured model {planner.model!r} is "
                    "not listed by GET /models; continuing because the catalog "
                    "may be dynamic"
                )
        except LLMPlanningError as error:
            print(
                "9ROUTER WARNING: GET /models advisory check failed; "
                f"the configured model will be tested by the real request: {error}"
            )
        print("9ROUTER PREFLIGHT: no quota-consuming startup requests")
        while rclpy.ok():
            command = input("\nEnter command (or 'quit'): ")
            if command.lower() in ("quit", "exit", "q"):
                break
            print(f"\nUSER COMMAND:\n{command}")
            try:
                print(f"\nLLM PLAN:\n{planner.plan(command)}")
            except LLMPlanningError as error:
                print(f"\nLLM ERROR: {error}")
    except LLMPlanningError as error:
        print(f"9ROUTER PREFLIGHT: FAILED ({error})")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
