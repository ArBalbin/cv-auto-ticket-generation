# 🎫 Queue Auto Ticket Generation using Computer Vision

![Python](https://img.shields.io/badge/Python-3.12-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Overview

The **Queue Auto Ticket Generation System** is an AI-powered solution that uses computer vision to detect when a person joins a queue and automatically generates and prints a ticket via a **thermal printer** — eliminating the need for manual ticket dispensing.

The system uses YOLOv8 to detect people in real-time from a camera feed. Once a person is detected entering the queue, a ticket is automatically issued and printed, streamlining the queuing process for public services, hospitals, government offices, and similar environments.

---

## ⚠️ Important: Files Not Included in This Repository

To keep this repository lightweight, the following large files have been **excluded** and must be downloaded separately before running the system:

### 1. 🤖 Trained Model Weights (`best.pt`)
Our custom-trained YOLOv8 model weights are stored on Google Drive.

📥 **Download here:** `[INSERT GOOGLE DRIVE LINK]`

After downloading, place the file here:
```
cv-auto-ticket-generation/
└── weights/
    └── best.pt
```

---

### 2. 📦 Base YOLOv8 Model (`yolov8n.pt`)
The base YOLOv8n model can be downloaded directly from Ultralytics:

📥 **Download here:** https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Place it in the root of the project:
```
cv-auto-ticket-generation/
└── yolov8n.pt
```

Or simply run the app — Ultralytics will auto-download it on first run.

---

### 3. 🗂️ Dataset (`data/`)
The training dataset (images + labels) is not included due to its large size.

📥 **Download here:** `[INSERT GOOGLE DRIVE LINK]`

After downloading and extracting, place the folder here:
```
cv-auto-ticket-generation/
└── data/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ArBalbin/cv-auto-ticket-generation.git
cd cv-auto-ticket-generation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
PRINTER_NAME=your_thermal_printer_name
```

### 4. Connect your Thermal Printer
- Install your thermal printer driver
- Make sure it is set as the **default printer** on your system
- Update `PRINTER_NAME` in your `.env` to match your printer's name

### 5. Download missing files
heres the link: 
Follow the **⚠️ Important** section above to download: 
- `best.pt` → place in `weights/`
- `yolov8n.pt` → place in root
- `data/` → place in root

### 6. Run the application
```bash
cd app
python main.py
```

Queue analytics is available at:
- Page: `http://localhost:5000/dashboard/queue-analytics`
- API: `http://localhost:5000/api/queue/analytics`

### Cloud deployment

For cloud deployment, use the checklist in `CLOUD_DEPLOYMENT_CHECKLIST.md`.

Minimum cloud start command:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

Set `PORTAL_BASE_URL` to your public cloud URL before generating tickets so QR codes point to the hosted backend. The detector should still run on the computer connected to the camera, with `API_BASE_URL` set to the cloud URL.

Ticket output note: the target deployment uses a thermal printer at the queue area, but this prototype currently generates PDF tickets as a temporary replacement while no thermal printer is available.

Staff registration is available at `/register` when `STAFF_REGISTRATION_ENABLED=1`. For cloud deployments, set `STAFF_REGISTRATION_CODE` so only authorized staff can create dashboard accounts.

For cloud MySQL, use `database/schema_cloud_ready.sql` for a fresh database. If your database already uses the older `queue_number INT NOT NULL UNIQUE` table, run `database/migrations/001_queue_records_cloud_ready.sql` as a MySQL admin/root user.

---

## 🔄 How It Works

```
Camera Feed
    ↓
YOLOv8 detects person entering queue
    ↓
System assigns queue number
    ↓
Ticket is auto-generated
    ↓
Thermal printer prints the ticket
    ↓
Person receives their queue number
```

---

## 📂 Project Structure

```
cv-auto-ticket-generation/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── detector.py              # Camera + YOLO detector process
│   ├── state.py                 # Shared API state
│   ├── core/
│   │   ├── config.py            # Environment/config values
│   │   └── security.py          # Auth/session/camera-token helpers
│   ├── database/
│   │   └── database_handler.py  # DB pool and DB helpers
│   ├── routers/
│   │   ├── auth.py              # /api/auth/*
│   │   ├── crowd.py             # /api/stats, snapshot, history
│   │   ├── detector_api.py      # /yolo/*
│   │   ├── health.py            # /health
│   │   ├── pages.py             # HTML dashboard routes
│   │   └── queue.py             # /api/queue/*
│   ├── services/
│   │   ├── prediction_service.py # Wait-time prediction helpers
│   │   ├── queue_service.py      # Queue API business logic
│   │   ├── queue_tracker.py      # Queue number tracking
│   │   ├── ticket_printer.py     # Ticket PDF/QR generation
│   │   └── ticket_service.py     # Background ticket worker
│   └── tickets/                 # Generated ticket PDFs
├── ML/                          # Model training files
├── Model/                       # Place model weights here
├── .env                         # Create this locally
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core programming language |
| YOLOv8 (Ultralytics) | Real-time person detection |
| OpenCV | Camera feed processing |
| Thermal Printer | Auto ticket printing |
| Arduino | IoT buzzer/alert integration |
| NumPy | Numerical operations |

---

## 👥 Team

| Name | GitHub |
|---|---|
| Archie Balbin | [@ArBalbin](https://github.com/ArBalbin) |
| *(more members coming soon)* | — |

---

## 📄 License

This project is licensed under the MIT License. See the [License](License) file for details.
