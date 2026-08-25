from pathlib import Path

import yaml


CONFIG_PATH = Path("config/models.yaml")


def load_models():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["models"]


def get_model_config(name):
    models = load_models()

    if name not in models:
        available = ", ".join(models.keys())

        raise ValueError(
            f"Unknown model: {name}\n"
            f"Available models: {available}"
        )

    return models[name]