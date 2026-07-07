def test_unrelated_documents_score_low():
    resume = "Experienced baker specializing in sourdough bread and pastry."
    jd = "Looking for a machine learning engineer skilled in Python and TensorFlow."
    score = compatibility_score(resume, jd)
    assert score < 20, f"Expected low score for unrelated docs, got {score}"


def test_extract_keywords_returns_list():
    jd = "We need a Python developer with Machine Learning and Data Science skills."
    keywords = extract_keywords(jd, top_n=5)
    assert isinstance(keywords, list)
    assert len(keywords) > 0


def test_analyze_returns_expected_keys():
    resume = "Python developer with Machine Learning experience."
    jd = "Looking for a Python and Machine Learning engineer."
    result = analyze(resume, jd)
    assert "score" in result
    assert "matched_keywords" in result
    assert "missing_keywords" in result
    assert 0 <= result["score"] <= 100


if _name_ == "_main_":
    tests = [
        test_clean_text_removes_punctuation_and_numbers,
        test_identical_documents_score_high,
        test_unrelated_documents_score_low,
        test_extract_keywords_returns_list,
        test_analyze_returns_expected_keys,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t._name_}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t._name_} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
