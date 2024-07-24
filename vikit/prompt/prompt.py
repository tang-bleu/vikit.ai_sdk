import os
from abc import ABC, abstractmethod
from typing import Any

import vikit.common.secrets as secrets
from vikit.prompt.prompt_build_settings import PromptBuildSettings


os.environ["REPLICATE_API_TOKEN"] = secrets.get_replicate_api_token()
"""
A subtitle file content looks lke this:

1
00:00:00,000 --> 00:00:05,020
I am losing my interest in human beings, in the significance of their lives and their actions.

2
00:00:06,080 --> 00:00:11,880
Someone has said it is better to study one man than ten books. I want neither books nor men,
"""


class Prompt(ABC):
    """
    A class to represent a prompt, a user written prompt, a prompt
    generated from an audio file, an image prompt, or one sent or received from an LLM.

    This class is going to be used as a base class for new type of prompts as
    they are accepted by LLM's, like a video, or an embedding...
    """

    def __init__(self):
        self.build_settings: PromptBuildSettings = PromptBuildSettings()
        self.title = "NoTitle"
        self._extended_fields: dict[str, Any] = {}

    @property
    def extended_fields(self) -> dict[str, Any]:
        return self._extended_fields

    @extended_fields.setter
    def extended_fields(self, value: dict[str, Any]):
        self._extended_fields = value
        if "title" in value:
            self.title = value["title"]
