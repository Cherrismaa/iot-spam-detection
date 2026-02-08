# 🧪 Efficient IoT Spam Detection

A full-stack Django web application that classifies **IoT messages** as *Spam* or *Normal* using machine learning. This project provides interactive prediction, model training, evaluation, and visualization features through web interfaces for users and administrators.

---

## 🚀 Project Overview

With the rapid growth of IoT devices, large volumes of machine-generated messages are produced. Many of these may be irrelevant or malicious. This system uses machine learning to automatically detect spam in IoT messages, helping improve data quality, system efficiency, and security.

This repository contains:

- A Django application with user and admin interfaces  
- Machine learning models trained on labeled dataset  
- Visualization charts for message type ratios and classifier performance  
- Downloadable prediction reports  

---

## 🛠 Built With

- **Django 6.0.2** – Web framework  
- **Pandas & NumPy** – Data processing  
- **scikit-learn** – Machine learning  
- **xlwt** – Export Excel reports  
- **HTML/CSS/JavaScript** – Frontend views  

---

## 📌 Key Features

✔ User registration and login  
✔ IoT message classification (Spam/Normal)  
✔ Multiple machine learning models  
✔ Admin dashboard with ratio and performance charts  
✔ Download predicted results as Excel  
✔ Training interface for models  

---

## 📈 Machine Learning Workflow

The application leverages labeled IoT message data for training and prediction:

1. Load **IOT_Datasets.csv**  
2. Map labels (`ham → 0`, `spam → 1`)  
3. Vectorize text using `CountVectorizer`  
4. Train multiple classifiers  
5. Evaluate models and store metrics  
6. Display prediction and performance in the UI

Models used:

- **Naive Bayes**
- **Support Vector Machine**
- **Logistic Regression**
- **Decision Tree**
- **Stochastic Gradient Descent (SGD)**

Metrics shown include accuracy, confusion matrix, and classification reports.

---

## 🧑‍💻 Getting Started

### Prerequisites

Make sure you have:

- Python 3.x installed  
- Virtual environment support  

---

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Cherrismaa/iot-spam-detection.git
   cd iot-spam-detection
   
2. Create and activate a virtual environment:

   ```bash 
   python -m venv venv
   venv\Scripts\activate

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt

4. Ensure the dataset file (IOT_Datasets.csv) is in the project root (next to manage.py).

5. Run migrations:

   ```bash# 🧪 Efficient IoT Spam Detection

A full-stack Django web application that classifies **IoT messages** as *Spam* or *Normal* using machine learning. This project provides interactive prediction, model training, evaluation, and visualization features through web interfaces for users and administrators.

---

## 🚀 Project Overview

With the rapid growth of IoT devices, large volumes of machine-generated messages are produced. Many of these may be irrelevant or malicious. This system uses machine learning to automatically detect spam in IoT messages, helping improve data quality, system efficiency, and security.

This repository contains:

- A Django application with user and admin interfaces  
- Machine learning models trained on a labeled dataset  
- Visualization charts for message type ratios and classifier performance  
- Downloadable prediction reports  

---

## 🛠 Built With

- **Django 6.0.2** – Web framework  
- **Pandas & NumPy** – Data processing  
- **scikit-learn** – Machine learning  
- **xlwt** – Export Excel reports  
- **HTML/CSS/JavaScript** – Frontend views  

---

## 📌 Key Features

✔ User registration and login  
✔ IoT message classification (Spam/Normal)  
✔ Multiple machine learning models  
✔ Admin dashboard with ratio and performance charts  
✔ Download predicted results as Excel  
✔ Training interface for models  

---

## 📈 Machine Learning Workflow

The application leverages labeled IoT message data for training and prediction:

1. Load **IOT_Datasets.csv**  
2. Map labels (`ham → 0`, `spam → 1`)  
3. Vectorize text using `CountVectorizer`  
4. Train multiple classifiers  
5. Evaluate models and store metrics  
6. Display prediction and performance in the UI

Models used:

- **Naive Bayes**
- **Support Vector Machine**
- **Logistic Regression**
- **Decision Tree**
- **Stochastic Gradient Descent (SGD)**

Metrics shown include accuracy, confusion matrix, and classification reports.

---

## 🧑‍💻 Getting Started

### Prerequisites

Make sure you have:

- Python 3.x installed  
- Virtual environment support  

---

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Cherrismaa/iot-spam-detection.git
   cd iot-spam-detection

   python manage.py migrate

6. Start the development server:

   ```bash
   python manage.py runserver

7. Open in your browser:

   ```bash
   http://127.0.0.1:8000/

---
## What you can do as an admin:

- Train machine learning models  
- View classification ratios  
- Download Excel reports  
- View prediction history  

---

## 🧠 Notes & Recommendations

- Do **not** place `nltk.download()` calls inside views — NLTK data should be downloaded once via a terminal session.
- For large datasets or production deployment, reuse trained models instead of retraining on every request.
- This project is intended for educational and demonstration purposes.
---
## 📌 Usage

### User Interface

- Register a new account  
- Log in and view your profile  
- Submit IoT messages for classification  

### Admin Interface

Log in with:  

 ```bash
Username: Admin
Password: Admin

---

##📫 Author

Cherrismaa
GitHub: https://github.com/Cherrismaa

Portfolio: https://cherrismaa.github.io/Cherrismaa-Portfolio/
