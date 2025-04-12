import joblib
import os


def inspect_model(model_path):
    """Inspect the contents of a model file."""
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if isinstance(model, dict):
        print("Dictionary keys:", model.keys())
        for key, value in model.items():
            print(f"- {key}: {type(value)}")

    return model


if __name__ == "__main__":
    model_path = "models/ensemble_H1_direction_12.joblib"
    model = inspect_model(model_path)