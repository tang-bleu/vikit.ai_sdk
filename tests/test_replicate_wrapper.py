from unittest.mock import patch
import pytest
from loguru import logger

from vikit.prompt.prompt_factory import PromptFactory
import vikit.gateways.ML_models_gateway_factory as ML_models_gateway_factory
from vikit.prompt.prompt_cleaning import cleanse_llm_keywords
from vikit.common.context_managers import WorkingFolderContext


SAMPLE_PROMPT_TEXT = """A group of ancient, moss-covered stones come to life in an abandoned forest, revealing intricate carvings
and symbols. This is additional text to make sure we generate serveral subtitles. """
ml_gw = ML_models_gateway_factory.MLModelsGatewayFactory().get_ml_models_gateway(
    test_mode=False
)


@pytest.mark.integration
def test_inte_get_keywords_from_prompt():
    with WorkingFolderContext():
        test_prompt = PromptFactory(ml_gateway=ml_gw).create_prompt_from_text(
            SAMPLE_PROMPT_TEXT
        )
        keywords, title = ml_gw.get_keywords_from_prompt(
            test_prompt.text, "previous_words"
        )
        assert len(keywords) > 0
        assert len(title) > 0


def test_extract_keywords_clean_nodigits():
    prompt = "A group of ancient, moss-covered stones come to life in an \n 8 abandoned forest, \n  revealing intricate,, carvings and symbols"

    result = cleanse_llm_keywords(prompt)
    assert not any(
        char.isdigit() for char in result
    ), "The result should not contain any digits"


def test_extract_keywords_clean_nodoublecomma():
    prompt = "A group of ancient, moss-covered stones come to life in an \n abandoned forest,, revealing intricate,, carvings and symbols"
    result = cleanse_llm_keywords(prompt)
    assert not result.__contains__(
        ",,"
    ), "The result should not contain ',,' i.e. double commas"


def test_extract_keywords_clean_nodots():
    prompt = "A group of ancient, moss-covered stones come to life in .  \n an abandoned forest, revealing intricate, carvings and symbols."
    result = cleanse_llm_keywords(prompt)
    assert not result.__contains__("."), "The result should not contain '.' i.e. dots"


def test_extract_keywords_clean_empty():
    prompt = ""
    result = cleanse_llm_keywords(prompt)
    logger.debug(f"res : {result}")
    assert result == "", "The result should be an empty string"


def test_extract_keywords_clean_None():
    with pytest.raises(AttributeError):
        prompt = None
        _ = cleanse_llm_keywords(prompt)
