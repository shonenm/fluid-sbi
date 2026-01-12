"""Tests for sda.config module."""

from __future__ import annotations

import pytest

from sda.config import (
    ExperimentConfig,
    ModelConfig,
    SDEConfig,
    Settings,
    TrainingConfig,
)


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.embedding == 64
        assert config.context == 0
        assert config.channels == [64, 128, 256]
        assert config.kernel_size == 3
        assert config.spatial == 2
        assert config.order == 1

    def test_custom_values(self):
        config = ModelConfig(embedding=128, channels=[32, 64])
        assert config.embedding == 128
        assert config.channels == [32, 64]

    def test_validation_embedding(self):
        with pytest.raises(ValueError):
            ModelConfig(embedding=0)

    def test_validation_channels(self):
        with pytest.raises(ValueError):
            ModelConfig(channels=[])


class TestSDEConfig:
    def test_default_values(self):
        config = SDEConfig()
        assert config.alpha == "cos"
        assert config.eta == 1e-3

    def test_valid_alpha_values(self):
        for alpha in ["lin", "cos", "exp"]:
            config = SDEConfig(alpha=alpha)
            assert config.alpha == alpha

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            SDEConfig(alpha="invalid")


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.epochs == 256
        assert config.batch_size == 64
        assert config.optimizer.learning_rate == 1e-3
        assert config.scheduler.name == "cosine"

    def test_nested_config(self):
        config = TrainingConfig(
            epochs=100,
            optimizer={"learning_rate": 1e-4, "weight_decay": 0.01},
        )
        assert config.epochs == 100
        assert config.optimizer.learning_rate == 1e-4
        assert config.optimizer.weight_decay == 0.01

    def test_validation_epochs(self):
        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)

    def test_validation_batch_size(self):
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=-1)


class TestExperimentConfig:
    def test_default_values(self):
        config = ExperimentConfig()
        assert config.seed == 42
        assert config.device == "cuda"
        assert config.debug is False

    def test_nested_config(self):
        config = ExperimentConfig(
            training={"epochs": 50},
            model={"embedding": 32},
            seed=123,
        )
        assert config.training.epochs == 50
        assert config.model.embedding == 32
        assert config.seed == 123

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            ExperimentConfig(unknown_field="value")

    def test_to_dict(self):
        config = ExperimentConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "training" in data
        assert "model" in data


class TestSettings:
    @pytest.mark.skipif(Settings is None, reason="pydantic-settings not installed")
    def test_default_values(self):
        settings = Settings()
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.json_logs is False

    @pytest.mark.skipif(Settings is None, reason="pydantic-settings not installed")
    def test_env_prefix(self):
        import os

        os.environ["SDA_DEBUG"] = "true"
        try:
            settings = Settings()
            assert settings.debug is True
        finally:
            os.environ.pop("SDA_DEBUG", None)
