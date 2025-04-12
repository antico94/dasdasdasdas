from typing import Dict, Optional, Type

from config.constants import ModelType
from models.base import BaseModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel


class ModelFactory:
    """Factory for creating model instances."""

    MODEL_REGISTRY = {
        ModelType.RANDOM_FOREST.value: RandomForestModel,
        ModelType.XGBOOST.value: XGBoostModel,
        ModelType.LSTM.value: LSTMModel,
        ModelType.ENSEMBLE.value: EnsembleModel
    }

    @classmethod
    def create_model(cls, model_type: str, config: Dict) -> BaseModel:
        """Create a model instance of the specified type."""
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls.MODEL_REGISTRY[model_type]
        return model_class(config)

    @classmethod
    def load_model(cls, model_path: str, model_type: Optional[str] = None) -> BaseModel:
        """Load a model from file."""
        if model_type is None:
            # Infer model type from filename
            if "random_forest" in model_path:
                model_type = ModelType.RANDOM_FOREST.value
            elif "xgboost" in model_path:
                model_type = ModelType.XGBOOST.value
            elif "lstm" in model_path:
                model_type = ModelType.LSTM.value
            else:
                model_type = ModelType.ENSEMBLE.value

        model = cls.create_model(model_type, {})
        model.load(model_path)
        return model