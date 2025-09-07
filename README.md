Redactor: Privacy-Safe Image Microservice

A lightweight microservice that detects faces and license plates in images and redacts them (blur or fill).
Built with FastAPI and OpenCV.

 Features
	•	Detects faces and license plates using Haar cascades.
	•	Two redaction modes:
	        Fill – cover with black rectangle.
	        Blur – Gaussian blur.
	•	REST API with interactive docs at /docs (Swagger UI).

How to Run
	1. Install the requirements:
		pip install -r requirements.txt

	2. Start the server:
		uvicorn app:app --reload

	3. Open Swagger UI

		Go to 👉 http://127.0.0.1:8000/docs

		Use /redact/ endpoint:

		Upload a JPEG or PNG image (< 5 MB)

		Choose mode (blur or fill)

		Get back the redacted image

		Use /delete/ endpoint:

		Delete previously uploaded files manually

Assumptions
	•	Only JPEG and PNG images are accepted.
	•	Max file size is 5 MB. Larger files are rejected.
	•	This service is for privacy redaction only — not general object detection.
	•	Output format matches the uploaded image (JPEG, PNG)

Testing
	Case 1: Valid image (< 5 MB)
		→ Faces/license plates are blurred or filled.
		→ If none found: {"message": "No face or license plate detected."}

	Case 2: Large image (> 5 MB)
		→ Service rejects with: "File too large. Max size = 5 MB"

Improvements / Next Steps
	•	Allowing multiple configurable redaction strategies (e.g. mosaic).
	•	Deploying on Render so others can test online.

Security 
	Malicious uploads
		Only JPEG/PNG are accepted, executables are rejected.

	Resource exhaustion
		File size limited to 5 MB.

	Auto-cleanup
		Uploaded files are deleted after 5 minutes.  


UML Class diagram
	The service follows an object-oriented design :
		- Detector→ handles face/license plate detection  
		- Strategy → defines redaction method (blur/fill)  
		- Filter → applies the selected redaction  
		- Controller →  API endpoints 


	[UML Diagram] 
