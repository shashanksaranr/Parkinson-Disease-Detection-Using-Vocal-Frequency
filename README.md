# Parkinson's Disease Detection Using Vocal Frequency

This project uses vocal frequency patterns for the detection of Parkinson's disease (PD). It employs machine learning techniques, including Deep Learning and Support Vector Machines (SVM), to classify speech recordings as indicative of PD or not. The system aims to provide early, non-invasive, and cost-effective detection for PD.

# 🚀 Features

Collects speech samples from both healthy individuals and Parkinson's patients.

Extracts vocal features like pitch, jitter, shimmer, and harmonic-to-noise ratio (HNR).

Trains a machine learning model to classify the voice recordings.

Provides a Streamlit web interface for easy testing.

Evaluates the model's performance with accuracy, precision, and recall metrics.

# 🛠️ Tech Stack

Backend: Python, PyTorch, Scikit-learn

Frontend: Streamlit

Machine Learning Algorithms: Deep Learning, SVM

Libraries: NumPy, Pandas, Matplotlib, Librosa (for feature extraction)

# ⚙️ Installation and Setup

1. Clone the repository:

    git clone https://github.com/your-username/parkinsons-disease-detection.git

    cd parkinsons-disease-detection

2. Create a virtual environment:

    python -m venv venv
   
    source venv/bin/activate   # For Linux/Mac
   
    venv\Scripts\activate      # For Windows
   
3. Install dependencies:
   
    pip install -r requirements.txt
   
4. Run the Streamlit application:

    streamlit run app.py
   
5. Access the application:
   
    Open http://localhost:8501 in your browser.
