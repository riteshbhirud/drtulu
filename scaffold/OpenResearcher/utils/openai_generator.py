"""
OpenAI API-compatible generator for agent inference
Works with vLLM OpenAI-compatible server or any OpenAI-compatible API
"""
from typing import List, Optional, AsyncIterator
import os
import httpx
import json


class ContextExhausted(Exception):
    """PATCHED (UMass ICLR runs): raised when the prompt has grown so large that
    no output tokens remain within the model's context window. The agent loop
    catches this and terminates the episode with terminated_reason
    'context_exhausted' rather than issuing a request that can only 400."""
    pass

# Pre-import transformers to avoid issues in multiprocessing
try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: transformers not available: {e}")
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


class OpenAIAsyncGenerator:
    """
    Async generator using OpenAI-compatible API
    Compatible with vLLM's OpenAI-compatible server
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = None,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        use_native_tools: bool = False
    ):
        """
        Args:
            base_url: Base URL of the OpenAI-compatible API (e.g., "http://localhost:8001/v1")
            model_name: Model name to use (if None, will use the default model from server)
            api_key: API key (use "EMPTY" for vLLM server)
            timeout: Request timeout in seconds
            use_native_tools: If True, use chat/completions API with native tools support
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.use_native_tools = use_native_tools
        self.client = httpx.AsyncClient(timeout=timeout)
        self._closed = False

        # Fetch tokenizer info from server
        self.tokenizer = None

    async def _init_tokenizer(self):
        """Initialize tokenizer by fetching model info from server"""
        if self.tokenizer is not None:
            return

        # Try to fetch model name if not set
        if not self.model_name:
            try:
                # Get model list to determine the actual model name
                response = await self.client.get(f"{self.base_url}/models")
                models_data = response.json()

                if models_data.get("data"):
                    # Use first available model
                    self.model_name = models_data["data"][0]["id"]
                    print(f"Auto-detected model: {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not fetch model list from server: {e}")

        # Use pre-imported AutoTokenizer if available
        if not _TRANSFORMERS_AVAILABLE or AutoTokenizer is None:
            raise ImportError("transformers library not available")

        # Try to load tokenizer from model name
        tokenizer_name = self.model_name
        if not tokenizer_name:
             raise ValueError("No model name provided and could not fetch from server")

        # Handle common model name prefixes
        if tokenizer_name.startswith("openai/"):
            tokenizer_name = tokenizer_name.replace("openai/", "OpenGPT-X/")

        # PATCHED (UMass ICLR runs): the model name sent to vLLM must match the
        # server's --served-model-name, but the tokenizer must load from a real
        # local path. Decouple them via TOKENIZER_PATH.
        _tok_override = os.environ.get("TOKENIZER_PATH", "").strip()
        if _tok_override:
            tokenizer_name = _tok_override

        print(f"Loading tokenizer for: {tokenizer_name}")
        # Direct load, no try/except fallback to Dummy
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        print(f"Tokenizer loaded successfully")

    async def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False
    ) -> AsyncIterator[int]:
        """
        Generate tokens using OpenAI API streaming

        Note: This method expects prompt_tokens but will decode them back to text
        for the API call (since OpenAI API uses text prompts)
        """
        await self._init_tokenizer()

        # Decode prompt tokens back to text
        # This is a workaround since OpenAI API expects text, not token IDs
        prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        # Prepare request
        # IMPORTANT: Don't pass stop strings to vLLM OpenAI API
        # Let the generation continue naturally and we'll check for stop conditions in the agent logic
        # PATCHED (UMass ICLR runs): the scaffold sent only `temperature`, so
        # every system ran with vLLM's defaults (top_p=1.0, top_k=-1,
        # presence_penalty=0.0) -- unconstrained sampling at temp 1.0. Each
        # system's published eval decoding config is supplied via env so runs
        # match the numbers those systems report.
        request_data = {
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": True,
            "temperature": float(os.environ.get("GEN_TEMPERATURE", temperature)),
            # Note: We intentionally don't set "stop" here to avoid premature stopping
            # The agent logic will check for <tool_response> and other markers
        }
        _top_p = os.environ.get("GEN_TOP_P", "").strip()
        if _top_p:
            request_data["top_p"] = float(_top_p)
        _pres = os.environ.get("GEN_PRESENCE_PENALTY", "").strip()
        if _pres:
            request_data["presence_penalty"] = float(_pres)
        _top_k = os.environ.get("GEN_TOP_K", "").strip()
        if _top_k:
            # top_k is a vLLM extension, not core OpenAI: send via extra body.
            request_data["top_k"] = int(_top_k)

        # PATCHED (UMass ICLR runs): clamp max_tokens against remaining context.
        # The original sent a fixed 8192 regardless of prompt length, so once a
        # trajectory exceeded (context_window - 8192) tokens every call returned
        # HTTP 400. _generate_with_retry then retried 20x, all failing identically,
        # and returned "" -- burning the rest of the turn budget on empty turns.
        # Observed in the DR-Venus smoke test: 41/641 calls (6.4%) failed this way.
        _ctx = int(os.environ.get("MODEL_CONTEXT_WINDOW", "131072"))
        _reserve = 256                      # headroom for template/special tokens
        _requested = max_tokens if (max_tokens and max_tokens > 0) else 8192
        _available = _ctx - len(prompt_tokens) - _reserve
        _final = max(0, min(_requested, _available))

        if _final <= 0:
            # No room left to generate: signal exhaustion instead of a doomed request.
            raise ContextExhausted(
                f"prompt={len(prompt_tokens)} tokens leaves no room in {_ctx}-token context"
            )
        request_data["max_tokens"] = _final

        if return_logprobs:
            request_data["logprobs"] = 1

        # Debug logging
        print(f"[OpenAI API] Request: model={self.model_name}, max_tokens={request_data['max_tokens']}")

        # Make streaming request
        async with self.client.stream(
            "POST",
            f"{self.base_url}/completions",
            json=request_data,
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])

                        if choices:
                            choice = choices[0]
                            text = choice.get("text", "")
                            finish_reason = choice.get("finish_reason")

                            if text:
                                # Encode text to tokens and yield
                                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                                for token in tokens:
                                    yield token

                            # Check for finish reason (only break on actual stop conditions)
                            # finish_reason can be: None (more coming), "stop" (stopped), "length" (max tokens)
                            if finish_reason is not None and finish_reason != "":
                                print(f"[OpenAI API] Stream finished with reason: {finish_reason}")
                                break

                    except json.JSONDecodeError:
                        continue

    async def chat_completion(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        use_reasoning_content: bool = True,
    ) -> dict:
        """
        Create a chat completion with optional tool calling using OpenAI Chat API

        Args:
            messages: List of message dicts with 'role' and 'content'/'reasoning_content'
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", or specific tool
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_reasoning_content: If True, use 'reasoning_content' field for assistant messages

        Returns:
            Response dict from API
        """
        await self._init_tokenizer()

        if not self.model_name:
            # Try to get model name from server
            try:
                response = await self.client.get(f"{self.base_url}/models")
                models_data = response.json()
                if models_data.get("data"):
                    self.model_name = models_data["data"][0]["id"]
            except:
                self.model_name = "unknown"

        # Convert messages to API format
        # Only keep valid fields and ensure content is never None
        api_messages = []
        for msg in messages:
            # Only keep these fields
            api_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content") or "",  # Ensure content is string, not None
            }

            # Add optional fields only if present
            if msg.get("reasoning_content"):
                api_msg["reasoning_content"] = msg["reasoning_content"]
            if msg.get("tool_calls"):
                api_msg["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                api_msg["tool_call_id"] = msg["tool_call_id"]

            api_messages.append(api_msg)

        request_data = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice

        print(f"[OpenAI Chat API] Request: model={self.model_name}, "
              f"messages={len(api_messages)}, tools={len(tools) if tools else 0}")
        print(f"[OpenAI Chat API] Request data: {json.dumps(request_data, indent=2, ensure_ascii=False)[:2000]}...")

        # Make request
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=request_data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # Print detailed error for 400 Bad Request
            print(f"[OpenAI Chat API] Error: {e.response.status_code} {e.response.reason_phrase}")
            print(f"[OpenAI Chat API] Response body: {e.response.text}")
            raise

    def shutdown(self) -> None:
        """Close the HTTP client"""
        if self._closed:
            return
        try:
            import asyncio
            asyncio.create_task(self.client.aclose())
        except Exception:
            pass
        finally:
            self._closed = True

    def __del__(self):
        try:
            if not getattr(self, "_closed", True):
                self.shutdown()
        except Exception:
            pass
