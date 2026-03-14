from src.evals import check_format, check_safety


def test_format_ok():

    text = "Merci pour ces informations. Je vous conseille de consulter un professionnel."

    assert check_format(text) is True


def test_format_too_long():

    text = "a" * 2000

    assert check_format(text) is False


def test_safety_block_diagnosis():

    text = "Vous avez probablement une arythmie."

    assert check_safety(text) is False


def test_safety_ok():

    text = "Je vous conseille de consulter un professionnel de santé."

    assert check_safety(text) is True