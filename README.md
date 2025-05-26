# Parkinson's Disease Detection Using Vocal Frequency

This project aims to detect Parkinson's Disease using vocal frequency features and machine learning models. Parkinson’s Disease (PD) is a neurodegenerative disorder that affects movement and speech. Vocal frequency characteristics can serve as a significant marker for early-stage PD diagnosis.

---

## 🧠 Project Overview

Parkinson’s Disease affects the vocal cords and speech production. By analyzing various vocal features extracted from patient voice samples, we can use machine learning techniques to classify whether a person has PD or not.

The system uses a dataset of biomedical voice measurements and builds a predictive model using algorithms like Support Vector Machines (SVM), Random Forest, and Artificial Neural Networks.

---

## 📂 Dataset

- **Name**: Parkinson's Telemonitoring Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Features**:
  - MDVP:Fo(Hz) – Average vocal fundamental frequency
  - MDVP:Jitter(%) – Variation in fundamental frequency
  - MDVP:Shimmer – Variation in amplitude
  - NHR, HNR – Noise to Harmonic Ratio
  - RPDE, DFA – Nonlinear dynamical complexity measures
  - Status – Health status (1 = Parkinson's, 0 = Healthy)

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Streamlit** (for UI)
- **Google Colab / Jupyter Notebook**
- **Snowflake** (for model storage and scaling) *(if applicable)*

---

## ⚙️ Installation

1. **Clone the repository**
   
   git clone https://github.com/yourusername/parkinsons-voice-detection.git
   cd parkinsons-voice-detection
   
2. Create a virtual environment (optional but recommended)

    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate

3. Install dependencies

    pip install -r requirements.txt
   
🚀 Usage

1. Training the Model
   
Run the following script to train the model:

    python train_model.py
    
2. Launching the Streamlit Web App

    streamlit run app.py
   
You can upload a CSV of vocal frequency features and get predictions directly through the interface.

📊 Model Performance

Model	Accuracy	Precision	Recall	F1-Score

Support Vector Machine	94.8%	0.95	0.93	0.94

Random Forest	96.2%	0.96	0.95	0.96

Neural Network	97.0%	0.97	0.96	0.96

The above results are subject to change based on hyperparameter tuning and dataset.

✅ Features

Input voice measurement features

Predict Parkinson’s status

Visualize feature importance

Streamlit web interface

Model training, evaluation, and export

Scalable backend using Snowflake (optional)


🔬 Future Enhancements

Integrate real-time voice recording and feature extraction

Deploy on cloud platforms (AWS/GCP)

Build an API for external use

Incorporate more diverse datasets for better generalization

📚 References

UCI Machine Learning Repository: Parkinson's Dataset

Kaggle Parkinson's Voice Dataset

Research papers on vocal biomarkers for Parkinson’s disease

🙋‍♂️ Author

SHASHANKSARAN R

GitHub: @shashanksaranr
