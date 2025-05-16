import os
from functools import cached_property
from typing import Dict, List, Optional, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger


@register_model(
    "deepseek-completions",
)
class DeepseekCompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.deepseek.com/v1/chat/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("DEEPSEEK_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `DEEPSEEK_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert (
            self.model
            in [
                "babbage-002",
                "davinci-002",
            ]
        ), f"Prompt loglikelihoods are only supported by Deepseek's API for {['babbage-002', 'davinci-002']}."
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("deepseek-chat-completions")
class DeepseekChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.deepseek.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("DEEPSEEK_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `DEEPSEEK_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as Deepseek does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert (
            type(messages) is not str
        ), "chat-completions require the --apply_chat_template flag."
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if "o1" in self.model:
            output.pop("stop")
            output["temperature"] = 1
        return output
