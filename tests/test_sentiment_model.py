import pytest

from src.predict import load_model, predict_texts


@pytest.fixture(scope="module")
def model():
    """Fixture to load the trained sentiment model once per test session."""
    model_path = "models/sentiment.joblib"
    return load_model(model_path)


@pytest.mark.parametrize(
    "input_text, expected_label",
    [
        ("I absolutely love this movie, it's fantastic!", 1),  # Positive
        ("The service was terrible and the food was disgusting.", 0),  # Negative
        ("This is a wonderful experience, I enjoyed every moment!", 1),  # Positive
        ("I hate this product, it was a complete waste of money.", 0),  # Negative
    ],
)
def test_sentiment_predictions(model, input_text, expected_label):
    """
    Test that the sentiment model predicts correct labels
    for obviously positive and negative sentences.
    """
    preds, probs = predict_texts(model, [input_text])

    # Check prediction result and type
    assert isinstance(preds[0], int), "Prediction should be an integer (0 or 1)."
    assert preds[0] in [0, 1], "Prediction should be 0 (negative) or 1 (positive)."
    assert preds[0] == expected_label, f"Expected {expected_label}, got {preds[0]}."

    # Optional sanity check: probability should be float between 0 and 1
    if probs[0] is not None:
        assert 0.0 <= probs[0] <= 1.0, "Probability must be between 0 and 1."
