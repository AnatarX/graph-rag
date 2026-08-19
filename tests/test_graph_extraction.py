import json

import pytest

from graph_rag.graph_extraction import _parse_json, is_valid_entity_name


def test_short_names_are_valid():
    assert is_valid_entity_name("Tony Blair")
    assert is_valid_entity_name("International Association of Athletics Federations")


def test_sentence_like_strings_are_rejected():
    assert not is_valid_entity_name(
        "52% rise in profits for the year to £198m from the £130m seen a year earlier"
    )
    assert not is_valid_entity_name("he probably has his own airplane seat that is how highly sony prize him")


def test_trailing_sentence_punctuation_is_rejected():
    assert not is_valid_entity_name("at least we are still talking.")


def test_empty_name_is_rejected():
    assert not is_valid_entity_name("")
    assert not is_valid_entity_name("   ")


def test_parse_json_valid_json_parses_as_is():
    raw = '{"entities": [{"name": "Tony Blair", "type": "person"}], "relations": []}'
    assert _parse_json(raw) == {
        "entities": [{"name": "Tony Blair", "type": "person"}],
        "relations": [],
    }


def test_parse_json_truncated_mid_string_raises_json_decode_error():
    # Оборвано посреди строкового значения, без закрывающей кавычки/скобки — ни
    # json.loads, ни regex-fallback (нужна закрывающая "}") с этим не справятся.
    # Это ОЖИДАЕМОЕ поведение: extract_from_text ловит именно json.JSONDecodeError
    # и логирует документ как "без сущностей", а не падает.
    raw = '{"entities": [{"name": "Tony Blair", "type": "per'
    with pytest.raises(json.JSONDecodeError):
        _parse_json(raw)


def test_parse_json_wrapped_in_extra_text_uses_regex_fallback():
    raw = 'Вот твой JSON: {"entities": [], "relations": []} Надеюсь, помогло!'
    assert _parse_json(raw) == {"entities": [], "relations": []}
