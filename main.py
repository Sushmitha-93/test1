from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# request schema


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default=None)
    num_images: Optional[int] = Field(default=1)
    width: Optional[int] = Field(default=1024)
    height: Optional[int] = Field(default=1024)
    steps: Optional[int] = Field(default=50)
    guidance_scale: Optional[float] = Field(default=7.5)

# Returns image


@app.post("/text-to-image", summary="Returns an image from a prompt")
async def text_to_image(data: TextToImageRequest):
    print("Prompt: ", data.prompt)
    image = pipe(prompt=data.prompt, negative_prompt=data.negative_prompt, height=data.height, width=data.width,
                 num_images_per_prompt=data.num_images, num_inference_steps=data.steps, guidance_scale=data.guidance_scale).images[0]

    # image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
