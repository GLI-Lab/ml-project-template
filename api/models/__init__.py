"""이미지 분류 모델 레지스트리.

새로운 모델을 추가하려면:
1. api/models/{model_name}/ 폴더를 만들고 base.ModelManager를 구현
2. 아래 model_managers 리스트에 등록
"""

import torchvision.models as models

from api.models.base import ModelManager
from api.models.resnet50.model import ResNet50Manager
from api.models.vit.model import ViTManager

model_managers: list[ModelManager] = [
    ResNet50Manager(
        name="resnet50",
        model_fn=models.resnet50,
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
    ),
    ViTManager(
        name="vit_b_16",
        model_fn=models.vit_b_16,
        weights=models.ViT_B_16_Weights.IMAGENET1K_V1,
    ),
]
