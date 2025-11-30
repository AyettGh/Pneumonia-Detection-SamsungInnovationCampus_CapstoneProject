# 🩺 Lung Infection Detection Model

This repository contains the code for an end-to-end AI system capable of detecting pneumonia from chest X-ray (CXR) images. It uses a YOLOv11 model, orchestrated with Vertex AI and MLflow, and served via a FastAPI backend to a Next.js frontend.

## 📂 Project Structure

```text
/lung-infection-detector
├── 📁 backend/                    # FastAPI Application
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 api.py                 # API router (e.g., /predict)
│   │   ├── 📄 model_client.py       # Client to call Vertex AI Endpoint
│   │   └── 📄 schemas.py            # Pydantic request/response models
│   ├── 📄 Dockerfile.api           # Dockerfile to deploy backend (e.g., to Cloud Run)
│   ├── 📄 main.py                 # Main FastAPI app entrypoint
│   └── 📄 requirements.txt        # Python deps (fastapi, uvicorn, google-cloud-aiplatform)
│
├── 📁 data/                       # Scripts for data handling (not for storing data!)
│   ├── 📄 01_download_data.py       # Script to fetch NIH dataset
│   ├── 📄 02_preprocess_dicom.py    # Script to convert DICOM to PNG/JPG
│   ├── 📄 03_prepare_yolo_labels.py # Script to convert annotations to YOLO format
│   └── 📄 04_upload_to_gcs.py       # Script to upload processed data to GCS for Vertex
│
├── 📁 frontend/                   # Next.js Application
│   ├── 📁 components/             # React components (e.g., Upload.tsx, Result.tsx)
│   ├── 📁 pages/                  # Next.js pages (index.tsx)
│   ├── 📁 services/               # API client (e.g., api.ts to call your FastAPI)
│   ├── 📄 Dockerfile.web           # Dockerfile for Next.js app
│   ├── 📄 next.config.js
│   └── 📄 package.json
│
├── 📁 model/                      # YOLOv11 Model Training
│   ├── 📁 config/
│   │   ├── 📄 data.yaml              # YOLO data config (paths to train/val in GCS)
│   │   └── 📄 yolov11.yaml            # YOLO model architecture config
│   ├── 📁 src/
│   │   ├── 📄 train.py                # Main training script (logs with MLflow)
│   │   ├── 📄 evaluate.py             # Evaluation script
│   │   └── 📄 export.py               # Script to export model to serving format
│   ├── 📄 Dockerfile.train          # Dockerfile for Vertex AI Custom Training
│   └── 📄 requirements.txt        # Python deps (ultralytics, mlflow, google-cloud-sdk)
├── 📁 notebooks/                  # Jupyter notebooks for exploration & R&D
│   ├── 📄 01-data-exploration.ipynb
│   └── 📄 02-prototype-model.ipynb
├── 📁 pipelines/                  # Vertex AI Pipelines (KFP)
│   ├── 📁 components/             # Reusable pipeline components (e.g., train, deploy)
│   │   ├── 📄 train_component.yaml
│   │   └── 📄 deploy_component.yaml
│   ├── 📄 pipeline.py               # Main KFP/Vertex AI pipeline definition
│   └── 📄 submit_pipeline.py        # Script to compile and submit the pipeline
│
├── 📄 .dockerignore
├── 📄 .gitignore                  # IMPORTANT: Ignore data, models, .env, node_modules
├── 📄 docker-compose.yml          # For local development (runs frontend, backend)
├── 📄 README.md                   # Project documentation
└── 📄 .env.example                # Template for environment variables
```

-----

## ⚙️ Workflow and Component Overview

This project is broken down into five core components, each with a specific responsibility.

### 1\. 📁 `data/`

  * **Purpose:** This folder contains scripts for data handling. It **does not** store the actual medical images, which should be in Google Cloud Storage (GCS).
  * **Workflow:**
    1.  **`01_download_data.py`**: Fetches the public NIH dataset.
    2.  **`02_preprocess_dicom.py`**: Converts the DICOM files into a web-friendly format like PNG or JPG.
    3.  **`03_prepare_yolo_labels.py`**: Reads the bounding box annotations and converts them into the `.txt` format that YOLO expects.
    4.  **`04_upload_to_gcs.py`**: Uploads the final `images/` and `labels/` folders to a GCS bucket so Vertex AI can access them for training.

### 2\. 📁 `model/`

  * **Purpose:** This is the heart of the AI. It contains everything needed to train the YOLOv11 model.
  * **Workflow:**
      * **`config/data.yaml`**: This file is configured to point to the `train/` and `val/` paths in your GCS bucket.
      * **`src/train.py`**: This is the main training script, integrated with MLflow.
          * Before training, the MLflow tracking URI is set to Vertex AI Experiments.
          * During training, it logs parameters (e.g., learning rate) using `mlflow.log_params()` and metrics (e.g., mAP, loss) using `mlflow.log_metrics()`.
      * **`Dockerfile.train`**: This crucial file packages the `model/` code and dependencies. Vertex AI Custom Training uses this Dockerfile to create and run a training job in the cloud.

### 3\. 📁 `pipelines/`

  * **Purpose:** This directory orchestrates the entire MLOps workflow using Vertex AI Pipelines.
  * **Workflow:**
      * **`pipeline.py`**: Defines the automated pipeline using the Kubeflow Pipelines (KFP) SDK.
      * This pipeline is a graph (DAG) of components, for example:
        1.  **Data Validation**: Checks if data in GCS is ready.
        2.  **Train Model**: Runs the `Dockerfile.train` (from `/model`) as a custom Vertex AI training job.
        3.  **Evaluate Model**: Checks if the model's mAP (logged via MLflow) is above a set threshold.
        4.  **Deploy Model**: If the evaluation is successful, automatically deploys the best model to a Vertex AI Endpoint.

### 4\. 📁 `backend/`

  * **Purpose:** A FastAPI server that acts as the "middle-man" between the frontend and the cloud-hosted AI model.
  * **Workflow:**
      * This server **does not** run the YOLO model itself, which is essential for scalability.
      * **`model_client.py`** uses the Google Cloud AI Platform SDK to communicate with the deployed model.
      * When a user uploads an image, the `/predict` endpoint in **`api.py`** will:
        1.  Receive the image.
        2.  Send the image to the production **Vertex AI Endpoint** for inference.
        3.  Receive the JSON response (bounding boxes, confidence scores).
        4.  Return this clean JSON to the frontend.
      * This service is containerized with **`Dockerfile.api`** and is designed to be deployed on a serverless platform like Google Cloud Run.

### 5\. 📁 `frontend/`

  * **Purpose:** The user interface, built in Next.js, that allows users to interact with the model.
  * **Workflow:**
    1.  A user visits the webpage (defined in `pages/index.tsx` or `app/page.tsx`).
    2.  They use the **`components/Upload.tsx`** component to upload a chest X-ray.
    3.  The frontend's API client (**`services/api.ts`**) sends this image to the FastAPI backend (`/backend`).
    4.  It receives the bounding box data as JSON from the backend.
    5.  The **`components/Result.tsx`** component displays the original image and draws the bounding boxes on top of it, likely using an HTML `<canvas>` element.

### 6\. 📁 `notebooks/`

  * **Purpose:** This is the "lab" or "scratchpad" for the project. It holds all Jupyter notebooks (.ipynb) used for data exploration, model prototyping, and testing small pieces of code before they are "graduated" into production scripts in the data/ or model/ folders.
  * **Note:** Code in this folder is for R&D and is not part of the production-deployed application.

-----

## 🚀 Getting Started

### Local Development

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/lung-infection-detector.git
    cd lung-infection-detector
    ```
2.  Set up environment variables:
    ```bash
    cp .env.example .env
    ```
3.  Fill in the required values in your new `.env` file (GCP project ID, bucket names, etc.).
4.  Build and run the local environment:
    ```bash
    docker-compose up --build
    ```
5.  Access the services:
      * **Frontend**: `http://localhost:3000`
      * **Backend API**: `http://localhost:8000/docs`#   P n e u m o n i a - D e t e c t i o n - S a m s u n g I n n o v a t i o n C a m p u s _ C a p s t o n e P r o j e c t  
 