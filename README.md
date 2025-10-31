```markdown
# CNN-based Indian Sign Language Detection

This project uses a 1D Convolutional Neural Network (CNN) with TensorFlow/Keras to detect static Indian Sign Language gestures (A, B, C) in real-time from a webcam.

This is a conversion and optimization of an original RNN-based project. The CNN architecture was chosen to better learn the spatial patterns of hand keypoints, and the data pipeline has been rebuilt from scratch for better performance and generalization.

![Demo Image](images/allGestures.png)

---

## 💡 How It Works

The model does **not** look at the raw video. Instead, it uses a highly efficient 2-step process:

1. **Hand Tracking (MediaPipe):** In real-time, Google's [MediaPipe](https://google.github.io/mediapipe/) scans the webcam feed to find the hand and extracts 42 keypoints (21 `x,y` coordinates).

2. **Preprocessing:** These 42 keypoints are processed to match the training data (`keypoint.csv`):
   - **Normalization (Part 1):** Keypoints are made relative to the wrist (landmark #0).
   - **Normalization (Part 2):** The entire pose is scaled between -1.0 and 1.0 based on its maximum keypoint value.

3. **Prediction (CNN):** This final `(1, 42)` array is fed into the trained `isl_cnn_model.h5`, which predicts the gesture.

---

## 🚀 How to Run This Project

You can train the model from scratch or use the included pre-trained `isl_cnn_model.h5` file.

### 1. Clone the Repository

Clone the repository and navigate into the folder:

```
git clone https://github.com/arduvey29/ISL-Detection-CNN.git
cd ISL-Detection-CNN
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

**Windows (Git Bash or CMD):**

```
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```
python -m venv venv
source venv/bin/activate
```

### 3. Install Libraries

Install all required libraries using the requirements.txt file:

```
pip install -r requirements.txt
```

### 4. Run the Live Detection

To run the real-time webcam detection using the pre-trained model:

```
python test_cnn.py
```

Show gestures 'A', 'B', or 'C' to the camera. It will show `...` if no hand is detected.

### 5. (Optional) Re-Train the Model

If you want to train the model yourself, just run the training script. This script automatically saves the best version of the model as `isl_cnn_model.h5` using EarlyStopping to prevent overfitting:

```
python train_cnn.py
```

---

## 📁 Project File Structure

```
project-root/
├── train_cnn.py                      # Script for training the CNN model
├── test_cnn.py                       # Script for live webcam detection
├── isl_cnn_model.h5                  # Pre-trained Keras model file
├── keypoint.csv                      # Dataset with 7,600+ hand poses
├── dataset_keypoint_generation.py    # Script to generate keypoint dataset
├── requirements.txt                  # Python dependencies
├── images/
│   └── allGestures.png              # Demo image for README
└── README.md                         # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `train_cnn.py` | Loads keypoint.csv, builds the CNN model, and saves `isl_cnn_model.h5` |
| `test_cnn.py` | Runs real-time webcam detection using MediaPipe and the trained model |
| `isl_cnn_model.h5` | Final trained Keras model file |
| `keypoint.csv` | Master dataset containing 7,600+ hand poses with labels |
| `dataset_keypoint_generation.py` | Original script to create keypoint.csv dataset |
| `requirements.txt` | List of all necessary Python libraries |

---

## 📋 Requirements

All dependencies are listed in `requirements.txt`. Key libraries include:

- TensorFlow/Keras (for CNN model)
- MediaPipe (for hand detection and keypoint extraction)
- OpenCV (for webcam and image processing)
- NumPy (for numerical computations)
- Pandas (for data handling)
- scikit-learn (for preprocessing)

---

## 🔧 Model Architecture

The CNN model is simple yet effective:

- **Input Layer:** Accepts 42 flattened keypoint values (21 landmarks × 2 coordinates)
- **Hidden Layer 1:** 64 units with ReLU activation + 50% Dropout
- **Hidden Layer 2:** 32 units with ReLU activation + 50% Dropout
- **Output Layer:** Softmax activation with number of classes (gestures)

The model uses:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping (patience=5) and ModelCheckpoint

---

## 📊 Performance

- **Test Accuracy:** Typically 85-95% depending on dataset quality
- **Real-Time Performance:** Runs smoothly on standard laptops with no GPU
- **Inference Time:** < 50ms per frame

---

## 🐙 Upload to GitHub

Follow these steps to upload your project to GitHub:

### Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and log in.
2. Click the **`+`** icon in the top-right corner and select **"New repository"**.
3. Give it a name (e.g., `CNN-Sign-Language-Detection`).
4. Click **"Create repository"**.
5. **Do NOT** check any boxes (like "Add a README" or "Add .gitignore"). You want an empty repository.

### Step 2: Push Your Project to GitHub

In your terminal, inside your project folder (where `train_cnn.py` is located):

```
git init
git add .
git commit -m "Initial commit: CNN-based Indian Sign Language Detection"
git branch -m main
git remote add origin https://github.com/YOUR_USERNAME/CNN-Sign-Language-Detection.git
git push -u origin main
```

### Step 3: Verify on GitHub

Refresh your GitHub repository page. Your project is now live with all files, README, and trained model.

---

## 🎯 Next Steps & Improvements

- Add support for more gestures (expand beyond A, B, C)
- Implement dynamic gesture recognition (sequences of gestures)
- Deploy as a web application using Flask/FastAPI
- Create a mobile app using TensorFlow Lite
- Add multi-hand detection support
- Improve robustness with data augmentation

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

---

## 📧 Contact

For questions or support, please contact the project maintainer or open an issue on GitHub.

---

**Happy coding! 🚀**
```