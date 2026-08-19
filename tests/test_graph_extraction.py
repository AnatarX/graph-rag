from graph_rag.graph_extraction import is_valid_entity_name


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
