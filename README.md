````md
# DarkIR 🔦  
Low-Light Image Enhancement API using PyTorch & FastAPI

DarkIR is a lightweight inference service built around the **DarkIR deep learning model** for low-light image enhancement.  
It exposes a simple **FastAPI** backend that can be run **locally** or **inside Docker**, and works across **macOS (ARM), Windows, and Linux**.

---

## 🚀 Features

- Low-light image enhancement using **DarkIR**
- FastAPI REST API
- CPU-only PyTorch (portable & stable)
- Multi-arch Docker support (`amd64` + `arm64`)
- Ready for production deployment

---

## 📁 Project Structure

```text
DarkIR/
├── archs/               # DarkIR model architecture
├── DarkIR_384.pt        # Pretrained model weights
├── main.py              # FastAPI application
├── requirements.txt     # Runtime dependencies
├── pyproject.toml       # Project metadata
├── Dockerfile           # Multi-arch Docker build
└── README.md
````

---

## 🧩 Requirements (Local)

* Python **3.12**
* pip
* macOS / Linux / Windows (CPU)

---

## 🔧 Install & Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/srvenu/DarkIR.git
cd DarkIR
```

---

### 2️⃣ Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ℹ️ PyTorch will be installed automatically based on your platform (CPU-only).

---

### 4️⃣ Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### 5️⃣ Open API docs

```
http://localhost:8000/docs
```

Available endpoints:

* `GET /` – service info
* `GET /health` – health check
* `POST /image-process` – image upload & enhancement

---

## 🐳 Run with Docker (Recommended)

### Prerequisites

* Docker Desktop (macOS / Windows)
* Docker Engine (Linux)

---

### 1️⃣ Pull the prebuilt image

```bash
docker pull srvenu/improve-image:latest
```

---

### 2️⃣ Run the container

```bash
docker run -it --rm \
  -p 8000:8000 \
  srvenu/improve-image:latest
```

Open:

```
http://localhost:8000/docs
```

---

## 🏗️ Build Docker Image Locally

### Single-arch (local testing)

```bash
docker build -t darkir:local .
```

Run:

```bash
docker run -p 8000:8000 darkir:local
```

---

### Multi-arch build (amd64 + arm64) with buildx

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t srvenu/improve-image:v0.1.0 \
  -t srvenu/improve-image:latest \
  --push .
```

> This produces one image that works on **Windows, macOS (Apple Silicon), and Linux**.

---

## 🧪 Example Request

Using `curl`:

```bash
curl -X POST "http://localhost:8000/image-process" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@input.jpg" \
  --output output.png
```

---

## ⚙️ Project Metadata

From `pyproject.toml`:

```toml
[project]
name = "darkir"
version = "0.1.0"
description = "DarkIR macOS ARM setup"
requires-python = ">=3.12,<3.13"
```

---

## 📌 Notes

* This project runs **CPU-only PyTorch** for portability.
* For GPU/CUDA builds, a separate Dockerfile is recommended.
* Model weights (`DarkIR_384.pt`) are loaded once at startup.

---

## 📄 License

This project is provided for research and development purposes.
Please check the original DarkIR paper/model license before commercial use.

---

## ✨ Author

**Venu Raj (srvenu)**
GitHub: [https://github.com/srvenu](https://github.com/srvenu)

```
