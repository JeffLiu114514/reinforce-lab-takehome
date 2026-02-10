from agent.extractor import _verify_snippet


def test_verification_exact():
    text = "Synthetic data can improve coverage."
    snippet = "Synthetic data can improve coverage."
    verified, method, score = _verify_snippet(text, snippet, threshold=85)
    assert verified is True
    assert method == "exact"
    assert score == 100.0


def test_verification_fuzzy():
    text = "Synthetic data can improve coverage in certain domains."
    snippet = "synthetic data improve coverage"
    verified, method, score = _verify_snippet(text, snippet, threshold=80)
    assert verified is True
    assert method == "fuzzy"
    assert score is not None
