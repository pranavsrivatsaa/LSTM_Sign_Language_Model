# 🤟 Real-Time Sign Language Detection & Translation

A full-stack computer vision and deep learning pipeline that captures sign language gestures via a webcam, extracts holistic skeletal landmarks, and translates them into text in real-time using a Recurrent Neural Network (LSTM) achieving an evaluation accuracy of **1.0**.

---

## 🚀 Features
* **Real-Time Translation:** Translates continuous sign language gestures ("hello", "thanks", "iloveyou") instantly on a live webcam feed.
* **Holistic Landmark Extraction:** Utilizes MediaPipe Holistic to track 1,662+ distinct spatial features across the face, pose, and both hands simultaneously.
* **Custom Dataset Collection:** Built with a specialized automated frame-collection pipeline featuring timed pauses to ensure clean action sequences and high data integrity.
* **High Accuracy Architecture:** Employs a multi-layer Long Short-Term Memory (LSTM) sequential deep learning model achieving a **1.0 accuracy score** on test datasets, deployed via pre-trained weights (`action.h5`).

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **TensorFlow / Keras** (LSTM Neural Network Architecture)
* **OpenCV** (Real-time video processing and UI rendering)
* **MediaPipe** (Holistic landmark tracking and spatial feature extraction)
* **Scikit-Learn & NumPy** (Data preprocessing, sequence splitting, and evaluation metrics)

---

## 🧠 System Architecture & Workflow

1. **Data Collection:** Python script records 30-frame video sequences per sign, capturing raw coordinate data using MediaPipe Holistic.
2. **Feature Extraction:** Extracts $x, y, z$ coordinates (plus visibility metrics) for 33 pose landmarks, 468 face landmarks, 21 left-hand landmarks, and 21 right-hand landmarks, flattening them into a dense 1,662-dimensional feature vector per frame.
3. **Model Training:** Feeds temporal sequences into a deep **Sequential LSTM model** optimized with categorical cross-entropy and the Adam optimizer over 2,000 epochs.
4. **Real-Time Inference:** Evaluates live video frames sequentially, mapping probabilities via a sliding window to render smooth, real-time translated text onto the screen.

---

## 📂 Repository Structure
```text
├── Sign_Language_Detection.ipynb   # Complete pipeline code (Collection, Training, & Evaluation)
├── action.h5                       # Pre-trained deep learning model weights
└── README.md                       # Project documentation
```
## ⚙️ How to Run Locally
Clone the repository:

Bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
Install dependencies:

Bash
pip install tensorflow opencv-python mediapipe scikit-learn numpy
Run the Live Test:
Open Sign_Language_Detection.ipynb in Jupyter Notebook and run the final cell to launch the real-time webcam translation feed!

## 🔍 Note on Generalization
While the model achieves an evaluation accuracy of 1.0 on the test split under controlled conditions, this proof-of-concept is optimized for a single user environment. Future iterations will focus on expanding the dataset with diverse signers and varied backgrounds to improve real-world robustness.
