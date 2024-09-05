from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from generators.utils import safe_format


class ImageTheme(BaseModel):
    """Base model for image generation themes."""

    model_config = ConfigDict(
        # Allow extra fields (e.g. for use with base classes)
        extra="allow")

    name: str = Field(description="The name of this theme")

    prompt_prefix: Optional[str] = Field(
        description=
        "Any extra words, including trigger words for LoRAs in this theme. Include a comma and spacing if you require it."
    )

    prompt_suffix: Optional[str] = Field(
        description=
        "Any extra words, including trigger words for LoRAs in this theme. Include a command and spacing if you require it."
    )

    model: str = Field(
        "stabilityai/stable-diffusion-xl-base-1.0",
        description=
        'Either (1) dall-e-3 or dall-e-2, or (2) URL or HuggingFace ID of the base model to generate the image. Examples: "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5", "SG161222/Realistic_Vision_V2.0". ',
    )
    custom_generate: bool = Field(
        False,
        description="Whether custom generation without themes"
    )
    def make_prompt(self, prompt: str, prompt_params: Optional[dict] = None):
        """Applies the included suffixes and then interpolates any {referenced} variables."""
        template = f"{self.prompt_prefix or ''} {prompt} {self.prompt_suffix or ''}"
        return safe_format(template, prompt_params or {})

    @property
    def is_dalle(self):
        return self.model in ["dall-e-2", "dall-e-3"]

    @property
    def is_custom_generator(self):
        return self.custom_generate



class DalleTheme(ImageTheme):
    """A Theme for a DALL-E model.

    This class is meant to completely capture a coherent set of generation config.

    The one thing it DOESN'T include is the fragment of "user prompt" that is custom to any one generation.

    This allows someone to separate:
     - the prompt (e.g. "a chest of pirate gold") from
     - the theme (model, style, quality, etc.)

     NOTE: DALL-E themes DO NOT support negative prompts. Any negative prompts will be ignored (currently)!
    """

    model: str = Field(
        "dall-e-3",
        description=
        "Model to use for image generation. Must be one of: ['dall-e-2', 'dall-e-3'].",
    )

    quality: str = Field(
        default="standard",
        description=
        "The quality of the image that will be generated. Must be one of: ['hd', 'standard']."
        "'hd' creates images with finer details and greater consistency across the image. "
        "This param is only supported for the `dall-e-3` model.",
    )

    style: str = Field(
        default="vivid",
        description=
        "The style of the generated images. Must be one of: ['vivid', 'natural']. "
        "Vivid causes the model to lean towards generating hyper-real and dramatic images. "
        "Natural causes the model to produce more natural, less hyper-real looking images. "
        "This param is only supported for `dall-e-3`.",
    )

    image_size: str = Field(
        "1024x1024",
        description=
        "The size of the generated image(s). For Dalle2: ['256x256', '512x512', '1024x1024']. For DALL-E 3: ['1024×1024', '1024×1792', '1792×1024']",
    )

    # TODO(dougreid): add validation for style and quality


class StableDiffusionTheme(ImageTheme):
    """A Theme for a StableDiffusion model.

    This class is meant to completely capture a coherent set of generation config.

    This allows someone to separate:
     - the prompt (e.g. "a chest of pirate gold") from
     - the theme (model, style, quality, etc.)

    The one thing it DOESN'T include is the fragment of "user prompt" that is custom to any one generation.

    This allows someone to separate:
     - the prompt (e.g. "a chest of pirate gold") from
     - the theme (SDXL w/ Lora 1, 2, a particula negative prompt addition, etc.
    """

    negative_prompt_prefix: Optional[str] = Field(
        description=
        "Any extra words, including trigger words for LoRAs in this theme. Include a comma and spacing if you require it."
    )

    negative_prompt_suffix: Optional[str] = Field(
        description=
        "Any extra words, including trigger words for LoRAs in this theme. Include a command and spacing if you require it."
    )

    model: str = Field(
        "stabilityai/stable-diffusion-xl-base-1.0",
        description=
        'URL or HuggingFace ID of the base model to generate the image. Examples: "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5", "SG161222/Realistic_Vision_V2.0". ',
    )

    loras: List[str] = Field(
        [],
        description=
        'The LoRAs to use for image generation. You can use any number of LoRAs and they will be merged together to generate the final image. MUST be specified as a json-serialized list that includes objects with the following params: - \'path\' (required)  - \'scale\' (optional, defaults to 1). Example: \'[{"path": "https://civitai.com/api/download/models/135931", "scale": 1}]',
    )

    seed: int = Field(
        -1,
        description=
        "The same seed and prompt passed to the same version of StableDiffusion will output the same image every time.",
    )

    image_size: str = Field(
        "square_hd",
        description=
        "The size of the generated image(s). You can choose between some presets or select custom height and width. Custom height and width values MUST be multiples of 8.Presets: ['square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 'landscape_4_3', 'landscape_16_9'] Custom Example: '{\"height\": 512,\"width\": 2048}'",
    )

    num_inference_steps: int = Field(
        30,
        description=
        "Increasing the number of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image.",
    )

    guidance_scale: float = Field(
        6,
        description=
        "The CFG(Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.",
    )

    clip_skip: int = Field(
        2,
        description=
        "Skips part of the image generation process, leading to slightly different results. This means the image renders faster, too.",
    )

    model_architecture: str = Field(
        "sdxl",
        description=
        "The architecture of the model to use. If a HF model is used, it will be automatically detected. Supported: ['sd', 'sdxl']",
    )

    scheduler: str = Field(
        "DPM++ 2M SDE",
        description=
        "Scheduler (or sampler) to use for the image denoising process. Possible values: ['DPM++ 2M', 'DPM++ 2M Karras', 'DPM++ 2M SDE', 'DPM++ 2M SDE Karras', 'Euler', 'Euler A']",
    )

    image_format: str = Field(
        "png",
        description=
        "The format of the generated image. Possible values: ['jpeg', 'png']",
    )

    def make_negative_prompt(self,
                             negative_prompt: str,
                             prompt_params: Optional[dict] = None):
        """Applies the included suffixes and then interpolates any {referenced} variables."""
        template = f"{self.negative_prompt_prefix or ''}{negative_prompt}{self.negative_prompt_suffix or ''}"
        return safe_format(template, prompt_params or {})
        
class CustomStableDiffusionTheme(StableDiffusionTheme):
    provider: str = Field(
        "fal_ai",
        description="Theme generation provider"
    )

class GetImgTheme(StableDiffusionTheme):
    width: int = Field(
        512,
        description="image width"
    )
    height: int = Field(
        768,
        description="image height"
    )
    num_inference_steps: int = Field(
        30,
        description=
        "Increasing the number of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image.",
    )

    guidance_scale: float = Field(
        6,
        description=
        "The CFG(Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.",
    )


class FluxTheme(StableDiffusionTheme):
    sync_mode: bool =Field(
        False,
        description="the function will wait for the image to be generated and uploaded before returning the response"

    )

FLUX_WITH_LORA = FluxTheme(
    name="flux_lora",
    model="flux-general-with-lora",
    image_size="portrait_4_3",
    prompt_prefix="",
    num_inference_steps=20,
    guidance_scale=4,
)

REALISTIC_VISION_V3 = GetImgTheme(
    name="realistic_vision_v3",
    model="realistic-vision-v3",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers,cloned face,cloned body",
    
)

ABSOLUTE_REALITY = GetImgTheme(
    name="absolute_reality_v1_8_1",
    model="absolute-reality-v1-8-1",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers,cloned face,cloned body",
    custom_generate=True,

)
DARK_SUSHI_MIX = GetImgTheme(
    name="dark_sushi_mix_v2_25",
    model="dark-sushi-mix-v2-25",
    #prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    custom_generate=True,
)
ARCANE_DIFFUSION = GetImgTheme(
    name="arcane_diffusion",
    model="arcane_diffusion",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    custom_generate=True,
)

VAN_GOGH_DIFFUSION = GetImgTheme(
    name="van_gogh_diffusion",
    model="van-gogh-diffusion",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    custom_generate=True,
)

NEVER_ENDING_DREAM = GetImgTheme(
    name="neverending_dream",
    model="neverending-dream",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    custom_generate=True,
)

MO_DI_DIFFUSION = GetImgTheme(
    name="mo_di_diffusion",
    model="mo-di-diffusion",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    custom_generate=True,
)


SYNTHWAVE_PUNK_V2 = GetImgTheme(
    name="synthwave_punk_v2",
    model="synthwave-punk-v2",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    custom_generate=True,
)

DREAM_SHAPER_v8 = GetImgTheme(
    name="dream_shaper_v8",
    model="dream-shaper-v8",
    negative_prompt_prefix="worst quality, low quality, normal quality, bad quality, poor quality, lowres, extra fingers, missing fingers, poorly rendered hands, mutation, deformed iris, deformed pupils, deformed limbs, missing limbs, amputee, amputated limbs",
    custom_generate=True,
)

BETTER_THAN_WORDS_SDXL_NSFW = CustomStableDiffusionTheme(
    name="better_than_words_sdxl_nsfw",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
    "https://civitai.com/api/download/models/233092?type=Model&format=SafeTensor&size=full&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",#'{\"height\": 1152,\"width\":896 }',#"portrait_4_3",
    custom_generate=True
)

LUSTIFY_SDXL_NSFW = CustomStableDiffusionTheme(
    name="lustify_sdxl_nsfw",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
    "https://civitai.com/api/download/models/233092?type=Model&format=SafeTensor&size=full&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",#'{\"height\": 1152,\"width\":896 }',#"portrait_4_3",
    custom_generate=True
)

SUZANNES_SDXL_NSFW = CustomStableDiffusionTheme(
    name="suzannesxl_nsfw",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
    "https://civitai.com/api/download/models/400093?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="Watermark, Text, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, , mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",#'{\"height\": 1152,\"width\":896 }',#"portrait_4_3",
    custom_generate=True
)

OMNIGEN_SDXL_NSFW = CustomStableDiffusionTheme(
    name="omnigen_sdxl_nsfw",
    #prompt_prefix="photorealistic, ",   
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
"https://civitai.com/api/download/models/228559?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",
    custom_generate=True
)

INIVERSE_MIX_SDXL_NSFW = CustomStableDiffusionTheme(
    name="iniverse_mix_sdxl_nsfw",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
    "https://civitai.com/api/download/models/294706",
    model_architecture="sdxl",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",
    custom_generate=True
)

ALBEDO_SDXL_NSFW = CustomStableDiffusionTheme(
    name="albedo_sdxl_nsfw",
    #prompt_prefix="photorealistic, ",
    #prompt_suffix=", realistic, highly detailed,highres, RAW,8k",
    model=
    "https://civitai.com/api/download/models/281176?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers",
    num_inference_steps=30,
    guidance_scale=5,
    clip_skip=2,
    scheduler="DPM++ 2M SDE",
    image_size="portrait_4_3",
    custom_generate=True)

ANYTHINGXL_SDXL_NSFW = CustomStableDiffusionTheme(
    name="anythingxl_sdxl_nsfw",
    prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/384264?type=Model&format=SafeTensor&size=full&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,airbrushed",
    num_inference_steps=28,
    guidance_scale=6,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

ANIMAGINE_SDXL_NSFW = CustomStableDiffusionTheme(
    name="animagine_sdxl_nsfw",
    #prompt_suffix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/293564?type=Model&format=SafeTensor&size=full&fp=fp32",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=6,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

CLEARHUNG_ANIME_SDXL_NSFW = CustomStableDiffusionTheme(
    name="clearhung_anime_sdxl_nsfw",
    #prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/156375",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=6,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

HASSAKU_SDXL_NSFW = CustomStableDiffusionTheme(
    name="hassaku_sdxl_nsfw",
    #prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/378499",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=6,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

ANIMEMIX_SDXL_NSFW = CustomStableDiffusionTheme(
    name="animemix_sdxl_nsfw",
    #prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/303526?type=Model&format=SafeTensor&size=full&fp=fp16",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=6,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

DEEPHENTAI_SDXL_NSFW = CustomStableDiffusionTheme(
    name="deephentai_sdxl_nsfw",
    #prompt_prefix="masterpiece, best quality, very aesthetic, absurdres, ",
    model=
    "https://civitai.com/api/download/models/286821",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=7,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

STABLEDIFFUSION_SDXL = CustomStableDiffusionTheme(
    name="stable_diffusion_1_5",
    #prompt_prefix="best quality, 4k, high resolution, photography, ",
    model=
    "stabilityai/stable-diffusion-xl-base-1.0",
    model_architecture="sdxl",
    negative_prompt_prefix="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,",
    num_inference_steps=28,
    guidance_scale=7,
    clip_skip=2,
    scheduler="Euler A",
    image_size="portrait_4_3",
    custom_generate=True)

REALISTIC_VISION = StableDiffusionTheme(
    name="realistic_vision",
    prompt_prefix="",
    model=
    "https://civitai.com/api/download/models/272376?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    model_architecture="sd",
    negative_prompt_prefix=
    "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
    num_inference_steps=30,
    guidance_scale=5,
    scheduler="DPM++ 2M SDE",
    clip_skip=2,
)


# Pixel Art XL (https://civitai.com/models/120096/pixel-art-xl) by https://civitai.com/user/NeriJS
PIXEL_ART_THEME_1 = StableDiffusionTheme(
    name="pixel_art_1",
    prompt_prefix="(pixel art) 16-bit retro, ",
    loras=["https://civitai.com/api/download/models/135931"],
)

# Pixel Art SDXL RW (https://civitai.com/models/114334/pixel-art-sdxl-rw) by https://civitai.com/user/leonnn1
PIXEL_ART_THEME_2 = StableDiffusionTheme(
    name="pixel_art_2",
    prompt_prefix="((pixelart)) 16-bit retro, ",
)

# From https://www.fal.ai/models/sd-loras
PIXEL_ART_THEME_3 = StableDiffusionTheme(
    name="pixel_art_3",
    prompt_prefix="isometric, 16-bit retro, ",
    negative_prompt_prefix=
    "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, ",
    model_architecture="sdxl",
    num_inference_steps=50,
    guidance_scale=7.5,
    clip_skip=0,
    loras=["https://civitai.com/api/download/models/130580"],
)

# From https://www.fal.ai/models/sd-loras
CINEMATIC_ANIMATION = StableDiffusionTheme(
    name="cinematic_animation",
    model="https://civitai.com/api/download/models/46846",
    prompt_prefix=
    "(masterpiece), (best quality), (incredible digital artwork), atmospheric scene inspired by a Peter Jackson fantasy movie, ",
    prompt_suffix=
    ", awe-inspiring structures, diverse and vibrant characters, engaging in a pivotal moment, dramatic lighting, vivid colors, intricate details, expertly capturing the essence of an epic cinematic experience <lora:epiNoiseoffset_v2Pynoise:1>",
    negative_prompt_prefix=
    "(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale), multiple views, comic, sketch, (((bad anatomy))), (((deformed))), (((disfigured))), watermark, multiple_views, mutation hands, mutation fingers, extra fingers, missing fingers, watermark, ",
    model_architecture="sd",
    num_inference_steps=90,
    guidance_scale=8,
    clip_skip=2,
)

# From https://www.fal.ai/models/sd-loras
FF7R = StableDiffusionTheme(
    name="ff7r",
    model="https://civitai.com/api/download/models/95489",
    prompt_prefix="ff7r style, blurry background, realistic, ",
    prompt_suffix=", ((masterpiece)) <lora:ff7r_style_ned_offset:1>",
    negative_prompt_prefix=
    "nsfw, (worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, (tan, muscular, loli, petite, child, infant, toddlers, chibi, sd character:1.1), multiple view, Reference sheet,",
    model_architecture="sd",
    num_inference_steps=80,
    guidance_scale=9,
    clip_skip=2,
    loras=["https://civitai.com/api/download/models/60948"],
)

# From https://www.fal.ai/models/sd-loras
EPIC_REALISM = StableDiffusionTheme(
    name="epic_realism",
    model="emilianJR/epiCRealism",
    prompt_prefix="photo, ",
    negative_prompt_prefix=
    "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
    model_architecture="sd",
    num_inference_steps=80,
    guidance_scale=5,
    clip_skip=0,
)

SD_XL_NO_LORAS = StableDiffusionTheme(
    name="stable_diffusion_xl_no_loras",
    negative_prompt_prefix=
    "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), (watermark), immature, child, ",
    model_architecture="sdxl",
)

DALL_E_3_VIVID_STANDARD = DalleTheme(
    name="dall_e_3_vivid_standard",
    model="dall-e-3",
    style="vivid",
    quality="standard",
)

DALL_E_3_VIVID_HD = DalleTheme(
    name="dall_e_3_vivid_hd",
    model="dall-e-3",
    style="vivid",
    quality="hd",
)

DALL_E_3_NATURAL_STANDARD = DalleTheme(
    name="dall_e_3_natural_standard",
    model="dall-e-3",
    style="natural",
    quality="standard",
)

DALL_E_3_NATURAL_HD = DalleTheme(
    name="dall_e_3_natural_hd",
    model="dall-e-3",
    style="natural",
    quality="hd",
)

DALL_E_2_STANDARD = DalleTheme(
    name="dall_e_2_standard",
    model="dall-e-2",
    quality="standard",
)

DALL_E_2_STELLAR_DREAMS = DalleTheme(
    name="dall_e_2_stellar_dream",
    model="dall-e-2",
    prompt_prefix="Surreal painting, ",
    prompt_suffix="; using soft, dreamy colors and elements of fantasy",
    quality="standard",
)

DALL_E_2_NEON_CYBERPUNK = DalleTheme(
    name="dall_e_2_neon_cyberpunk",
    model="dall-e-2",
    prompt_prefix="Cyberpunk, digital art, best quality, neon-lit, ",
    prompt_suffix="; high contrast, blending traditional and futuristic",
    quality="standard",
)

# Premade themes that we know work well
PREMADE_THEMES = [
    PIXEL_ART_THEME_1, PIXEL_ART_THEME_2, PIXEL_ART_THEME_3, FF7R,
    CINEMATIC_ANIMATION, EPIC_REALISM, SD_XL_NO_LORAS, DALL_E_3_NATURAL_HD,
    DALL_E_3_NATURAL_STANDARD, DALL_E_3_VIVID_HD, DALL_E_3_VIVID_STANDARD,
    DALL_E_2_STANDARD, DALL_E_2_STELLAR_DREAMS, DALL_E_2_NEON_CYBERPUNK,
    REALISTIC_VISION, 
    OMNIGEN_SDXL_NSFW, 
    CLEARHUNG_ANIME_SDXL_NSFW,
    ANIMAGINE_SDXL_NSFW,
    ANIMEMIX_SDXL_NSFW,
    ALBEDO_SDXL_NSFW,
    INIVERSE_MIX_SDXL_NSFW,
    HASSAKU_SDXL_NSFW,
    DEEPHENTAI_SDXL_NSFW,
    ANYTHINGXL_SDXL_NSFW,
    REALISTIC_VISION_V3,
    BETTER_THAN_WORDS_SDXL_NSFW,
    FLUX_WITH_LORA,
    ABSOLUTE_REALITY,
    DARK_SUSHI_MIX,
    ARCANE_DIFFUSION,
    NEVER_ENDING_DREAM,
    MO_DI_DIFFUSION,
    SYNTHWAVE_PUNK_V2,
    DREAM_SHAPER_v8,
    VAN_GOGH_DIFFUSION,
    STABLEDIFFUSION_SDXL,
    SUZANNES_SDXL_NSFW,
    LUSTIFY_SDXL_NSFW
]

DEFAULT_THEME = ALBEDO_SDXL_NSFW
