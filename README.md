# Redactor: Privacy-Safe Image Microservice

A *lightweight microservice* that detects faces and license plates in images and redacts them (blur or fill).  
Built with *FastAPI* and *OpenCV*.

---

## Features
- *Detects faces and license plates* using Haar cascades.  
- *Two redaction modes:*  
  - *Fill* – cover with black rectangle  
  - *Blur* – Gaussian blur  
- *REST API* with interactive docs at /docs (Swagger UI)  

---

## How to Run
- *Install the requirements:*  
```bash
pip install -r requirements.txt
```

- *Start the server:*
```bash
uvicorn app:app --reload
```

- *Open Swagger UI:*  
http://127.0.0.1:8000/docs  

---

## API Endpoints

| Endpoint      | Method | Description |
|--------------|--------|-------------|
| /redact/   | POST   | Upload a JPEG/PNG image (< 5 MB), select mode (blur or fill), and receive the redacted image in response. |
| /delete/   | DELETE | Manually delete previously uploaded files from the server. |

---

## Assumptions

- Only JPEG and PNG images are accepted.
- Max file size = 5 MB (larger files are rejected).
- This service is for privacy redaction only – not general object detection.
- Output format matches the uploaded image (JPEG, PNG).

---

## Testing

*Case 1:* Valid image (< 5 MB)<br>
Faces/license plates are blurred or filled.<br>
If none found:<br>
json
{"message": "No face or license plate detected."}


*Case 2:* Large image (> 5 MB)<br>
Service rejects with:<br>

"File too large. Max size = 5 MB"


---

## Improvements / Next Steps:
- Allow multiple configurable redaction strategies (e.g., mosaic).  
- Deploy on Render so others can test online.

---

## Security

- *Malicious uploads:* Only JPEG/PNG are accepted; executables are rejected.  
- *Resource exhaustion:* File size limited to 5 MB.  
- *Auto-cleanup:* Uploaded files are deleted after 5 minutes.  

---

## UML Class Diagram

The service follows an object-oriented design:

- *Detector* → handles face/license plate detection  
- *Strategy* → defines redaction method (blur/fill)  
- *Filter* → applies the selected redaction  
- *Controller* → API endpoints  
  ![UML Diagram](https://github.com/sannshaikh23/Redactor-service/blob/main/UML%20diagram.jpg)
