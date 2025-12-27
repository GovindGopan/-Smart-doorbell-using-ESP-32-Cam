# Smart Doorbell Backend

This project is the backend for a smart doorbell system that uses AI to detect people, recognize faces, and detect packages. It sends email alerts and can trigger a buzzer on an ESP32 device.

## Features

*   **Person Detection:** Detects if a person is at the door using the YOLOv8 model.
*   **Face Recognition:** Recognizes known faces using MTCNN and FaceNet.
*   **Package Detection:** Detects if a package has been left at the door using a custom-trained YOLOv8 model.
*   **Email Alerts:** Sends email notifications with a snapshot of the person or package detected.
*   **ESP32 Integration:** Triggers a buzzer on an ESP32 device when a person or package is detected.
*   **FastAPI Backend:** Built with FastAPI, providing a robust and easy-to-use API.

## Getting Started

### Prerequisites

*   Python 3.8+
*   An ESP32 device flashed with the appropriate firmware to handle buzzer activation.
*   A Gmail account with an "App Password" for sending email alerts.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the root of the project and add the following variables:

    ```
    EMAIL_ID="your_gmail_id@gmail.com"
    EMAIL_PASSWORD="your_gmail_app_password"
    ESP32_IP="http://your_esp32_ip_address"
    ```

    Alternatively, you can set these as system environment variables.

5.  **Download the model weights:**

    The `.gitignore` file is configured to exclude the model weights. You will need to download the following pre-trained models and place them in the root of the project directory:

    *   `yolov8n.pt`: [https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
    *   `best.pt`: This is your custom-trained model for package detection. You will need to provide this file.
    *   `known_faces.pt`: This file contains the embeddings of known faces. You can generate this file by running the `train_known_faces.py` script.

### Running the Application

To start the FastAPI server, run the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`.

## API Endpoints

*   `POST /upload`: Upload an image to be processed by the AI models.
*   `POST /setup-esp32-ip`: Set the IP address of the ESP32 device.
*   `POST /esp32-log`: Log messages from the ESP32.
*   `POST /buzzer-log`: Log when the IR buzzer is activated.

## How to Train the Face Recognition Model

1.  Place images of known individuals in the `images` folder. Each person should have their own subfolder named after them (e.g., `images/John Doe/`).
2.  Run the `train_known_faces.py` script:
    ```bash
    python train_known_faces.py
    ```
    This will generate the `known_faces.pt` file.

---

_This project was created with the assistance of an AI agent._
