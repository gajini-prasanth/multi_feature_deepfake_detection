from .fusion_model import MultiFeatureFusionModel


class DeepfakeDetector(MultiFeatureFusionModel):
    """Track B model: Deepfake vs. real face frame detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
