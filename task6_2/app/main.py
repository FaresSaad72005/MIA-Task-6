from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from io import BytesIO
import sys
from pathlib import Path

# Add the root directory (task6_2) to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.deployment import deploy

# ---------------------------------------------------------------------------------------------
# cd /mnt/e/linux_projects/MIA/task6/task6_2
# /mnt/e/linux_projects/.venv/bin/uvicorn app.main:app --reload
# ---------------------------------------------------------------------------------------------
# cd /mnt/e/linux_projects/MIA/task6/task6_2/app
# /mnt/e/linux_projects/.venv/bin/uvicorn main:app --reload
# ---------------------------------------------------------------------------------------------
# /mnt/e/linux_projects/MIA/task6/task6_2$ docker run --rm -p 8000:8000 -p 7861:7860 image-captioning

# Create FastAPI application
app = FastAPI( title="Image Captioning API",
              description="API for generating captions from images using an attention-based image captioning model.",
              version="1.0.0")


# Home endpoint
@app.get("/")
def home():
    return { "message": "Image Captioning API is running"}


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Image caption endpoint
@app.post("/caption")
async def generate_caption(image: UploadFile = File(...)):

    # Check the uploaded file is an image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException( status_code=400, detail="The uploaded file must be an image.")

    # Read uploaded image
    image_bytes = await image.read()

    # Convert bytes to PIL Image
    try:
        pil_image = Image.open( BytesIO(image_bytes)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")


    # Generate caption using your ML function
    try:
        caption = deploy( pil_image )

    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Caption generation failed: {str(e)}")


    return {"filename": image.filename, "caption": caption}