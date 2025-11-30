1. Backend (FastAPI)
/lung-infection-detector/backend/ # FastAPI backend

app/

|— init.py

|— api.py # API routes

|— model_client.py # Vertex AI client for model serving

|— schemas.py # Pydantic request/response models

Dockerfile.api # Dockerfile for backend deployment

main.py # FastAPI app entrypoint

requirements.txt # Python dependencies

2. Data Preparation
|— data/ # Data preparation scripts

|— 01_download_data.py

|— 02_preprocess_dicom.py

|— 03_prepare_yolo_labels.py

|— 04_upload_to_gcs.py

3. Frontend (Next.js)
|— frontend/ # Next.js frontend

components/ # React components (e.g., Upload.tsx, Result.tsx)

pages/ # Next.js pages (e.g., index.tsx)

services/ # API client (e.g., api.ts)

Dockerfile.web

next.config.js

package.json # Node.js dependencies

4. Model & Training
|— model/ # YOLOv11 model training

config/

|— data.yaml

|— yolov11.yaml

src/

|— train.py

|— evaluate.py

|— export.py

Dockerfile.train

requirements.txt # Python dependencies for training

|— notebooks/ # Exploration & prototyping notebooks (KFP)

|— 01-data-exploration.ipynb

|— 02-prototype-model.ipynb

5. MLOps Pipelines (Vertex AI/KFP)
|— pipelines/ # Vertex AI pipelines (KFP)

components/

|— train_component.yaml

|— deploy_component.yaml

|— pipeline.py # Main pipeline definition

6. Root/Configuration Files
.dockerignore

.gitignore

docker-compose.yaml # Orchestration for multi-service setup

README.md # This file!

.env.example # Template for environment variables
