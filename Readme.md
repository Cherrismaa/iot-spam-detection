# IoT Spam Detection using Machine Learning

A Django-based web application that classifies IoT-generated messages as **Spam** or **Normal** using Natural Language Processing and Ensemble Machine Learning techniques.

The system integrates multiple classifiers and applies a voting mechanism to improve prediction reliability within a full-stack implementation.

---

## Tech Stack

- **Backend:** Django (Python)  
- **Machine Learning:** Scikit-learn  
- **Text Processing:** NLTK, CountVectorizer  
- **Data Handling:** Pandas, NumPy  
- **Frontend:** HTML, CSS (Django Templates)  
- **Database:** SQLite  

---

## ML Models Used

- Multinomial Naive Bayes  
- Support Vector Machine (LinearSVC)  
- Logistic Regression  
- Decision Tree  
- Voting Classifier (Ensemble Model)  

---

## Features

- User authentication system
- IoT message submission interface
- Real-time spam prediction
- Ensemble-based classification
- Prediction history storage
 
## Future Enhancements
 
- Persist trained model instead of training per request
- Convert to REST API
- Dockerize application
- Deploy to a cloud platform

## Installation & Setup

```bash
git clone https://github.com/Cherrismaa/iot-spam-detection.git
cd efficient-spam-detection

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver