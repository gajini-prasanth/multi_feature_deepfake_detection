from .fusion_model import MultiFeatureFusionModel


class AIImageDetector(MultiFeatureFusionModel):
    """Track A model: AI-generated vs. real photograph detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
