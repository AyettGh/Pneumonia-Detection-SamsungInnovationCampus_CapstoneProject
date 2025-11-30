/lung-infection-detector
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api.py            # API routes
│   │   ├── model_client.py   # Vertex AI client
│   │   └── schemas.py        # Pydantic request/response models
│   ├── Dockerfile.api        # Dockerfile for backend deployment
│   ├── main.py               # FastAPI app entrypoint
│   └── requirements.txt
│
├── data/                     # Data preparation scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess_dicom.py
│   ├── 03_prepare_yolo_labels.py
│   └── 04_upload_to_gcs.py
│
├── frontend/                 # Next.js frontend
│   ├── components/           # React components (Upload.tsx, Result.tsx)
│   ├── pages/                # Next.js pages (index.tsx)
│   ├── services/             # API client (api.ts)
│   ├── Dockerfile.web
│   ├── next.config.js
│   └── package.json
│
├── model/                    # YOLOv11 model training
│   ├── config/
│   │   ├── data.yaml
│   │   └── yolov11.yaml
│   ├── src/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── export.py
│   ├── Dockerfile.train
│   └── requirements.txt
│
├── notebooks/                # Exploration & prototyping notebooks
│   ├── 01-data-exploration.ipynb
│   └── 02-prototype-model.ipynb
│
├── pipelines/                # Vertex AI pipelines (KFP)
│   ├── components/
│   │   ├── train_component.yaml
│   │   └── deploy_component.yaml
│   ├── pipeline.py
│   └── submit_pipeline.py
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── README.md
└── .env.example
