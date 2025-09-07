from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import threading
import time
from redactor import Redactor

# ---------------- App Setup ---------------- #
app = FastAPI(
    title="Redactor Web Service",
    description="Redact faces and license plates from images",
    version="1.0"
)
redactor = Redactor()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

ALLOWED_TYPES = ["image/jpeg", "image/png"]

# ---------------- Background cleanup ---------------- #
def schedule_file_cleanup(files, delay=300):
    """Delete files after a delay (default 5 min)."""
    def cleanup():
        time.sleep(delay)
        for f in files:
            if os.path.exists(f):
                os.remove(f)
                print(f"[CLEANUP] Deleted {f}")
    threading.Thread(target=cleanup, daemon=True).start()


# ---------------- API Routes ---------------- #

@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok"}


@app.post("/redact/")
async def redact_image(file: UploadFile = File(...), mode: str = Form("blur")):
    """
    Upload an image and redact faces/license plates.
    Returns the redacted image as a file download.
    """
    try:
        # ---- Check MIME type ----
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are allowed."
            )

        # ---- Check file size ----
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File too large. Max size = 5 MB"
            )
        await file.seek(0)  # reset pointer

        # ---- File paths ----
        ext = ".png" if file.content_type == "image/png" else ".jpg"
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(UPLOAD_FOLDER, f"redacted_{file.filename}")

        # ---- Save uploaded file ----
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---- Run redaction ----
        detected = redactor.redact(input_path, output_path, mode=mode)

        # ---- Auto cleanup ----
        schedule_file_cleanup([input_path, output_path], delay=300)

        # ---- If nothing detected ----
        if not detected:
            return JSONResponse({"message": "No face or license plate detected."})

        # ---- Return file ----
        return FileResponse(
            output_path,
            media_type=file.content_type,
            filename=f"redacted_{file.filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete/")
async def delete_files(files: str = Form(...)):
    """
    Delete uploaded files manually (comma-separated filenames).
    Only deletes files inside the uploads/ folder.
    """
    deleted = []
    for fname in files.split(","):
        f = os.path.join(UPLOAD_FOLDER, os.path.basename(fname.strip()))
        if os.path.exists(f):
            os.remove(f)
            deleted.append(f)

    return {"deleted": deleted}
