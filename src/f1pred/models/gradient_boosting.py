from sklearn.ensemble import GradientBoostingClassifier

def build(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
):
    """Return an unfitted GradientBoostingClassifier with sensible defaults."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

# convenience alias if you prefer this name
get_model = build