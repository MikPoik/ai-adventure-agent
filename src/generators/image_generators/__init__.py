from generators.image_generator import ImageGenerator
from generators.image_generators.dalle import DalleImageGenerator
from generators.image_generators.flux_with_loras import FluxImageGenerator
from generators.image_generators.stable_diffusion_with_loras import (
    StableDiffusionWithLorasImageGenerator,
)
from generators.image_generators.custom_stable_diffusion_with_loras import (
    CustomStableDiffusionWithLorasImageGenerator,
)
from generators.image_generators.get_img_ai import (
    GetimgAiImageGenerator,
)
from generators.image_generators.flux_with_loras import (
    FluxImageGenerator,
)
from schema.image_theme import CustomStableDiffusionTheme, FluxTheme, GetImgTheme, ImageTheme
import logging

def get_image_generator(theme: ImageTheme) -> ImageGenerator:
    if theme.is_dalle:
        return DalleImageGenerator()
    elif isinstance(theme, CustomStableDiffusionTheme):
        return CustomStableDiffusionWithLorasImageGenerator()
    elif isinstance(theme, GetImgTheme):
        return GetimgAiImageGenerator()
    elif isinstance(theme, FluxTheme):
        return FluxImageGenerator()
    return StableDiffusionWithLorasImageGenerator()
