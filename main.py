from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from archs import DarkIR

# ============================================================
# App
# ============================================================
app = FastAPI(
    title="DarkIR Image Processing API",
    version="1.0.0",
    description="Low-light image enhancement using DarkIR"
)

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Transforms
# ============================================================
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

# ============================================================
# Model Loading (ONCE at startup)
# ============================================================
MODEL_PATH = "DarkIR_384.pt"

model = DarkIR(
    img_channel=3,
    width=32,
    middle_blk_num_enc=2,
    middle_blk_num_dec=2,
    enc_blk_nums=[1, 2, 3],
    dec_blk_nums=[3, 1, 1],
    dilations=[1, 4, 9],
    extra_depth_wise=True,
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["params"])
model.to(device)
model.eval()

# ============================================================
# Helpers
# ============================================================
def pad_tensor(tensor, multiple: int = 8):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=0)

@torch.no_grad()
def run_inference(image: Image.Image) -> Image.Image:
    tensor = pil_to_tensor(image).unsqueeze(0).to(device)
    _, _, h, w = tensor.shape

    tensor = pad_tensor(tensor)

    output = model(tensor, side_loss=False)
    output = torch.clamp(output, 0.0, 1.0)
    output = output[:, :, :h, :w].squeeze(0)

    return tensor_to_pil(output)

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    """Service info endpoint."""
    return {
        "service": "DarkIR Image Processing API",
        "status": "running",
        "device": str(device),
    }

@app.get("/health")
def health():
    """Health check endpoint (Docker / K8s friendly)."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": True,
            "device": str(device),
        },
    )

@app.post("/image-process")
async def image_process(file: UploadFile = File(...)):
    """
    Upload an image and receive enhanced output.
    Testable directly from /docs
    """
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    output_image = run_inference(image)

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=output.png"},
    )
