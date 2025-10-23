# 🎬 Personalized Movie Recommendation Engine using Spark MLlib and Flask

## 📘 Overview
This project implements a **personalized movie recommendation system** similar to Netflix or Amazon using **Apache Spark MLlib’s ALS (Alternating Least Squares)** algorithm.  
It predicts user preferences based on their historical movie ratings and provides **top-5 personalized movie recommendations** through a Flask REST API.

---

## ⚙️ Tech Stack
| Technology | Purpose |
|-------------|----------|
| **Apache Spark (MLlib)** | Distributed processing & collaborative filtering |
| **Python (PySpark)** | Programming interface for Spark |
| **Flask** | Web framework for deployment |
| **Pandas, NumPy** | Data preprocessing & transformation |
| **MovieLens 100k Dataset** | Dataset for model training/testing |
| **Linux (Arch)** | Development environment |

---

## 🧠 Project Workflow

1. **Data Collection**  
   Downloaded the MovieLens 100k dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k).

2. **Data Preprocessing**  
   Loaded the dataset into Spark DataFrame, renamed columns, and split it into 80% training and 20% testing.

3. **Model Building**  
   Implemented the **ALS (Alternating Least Squares)** algorithm using Spark MLlib to predict unseen ratings.

4. **Model Evaluation**  
   Calculated the **Root Mean Square Error (RMSE)** to evaluate prediction accuracy.

5. **Model Saving**  
   Saved the trained model for later reuse in deployment.

6. **Deployment**  
   Created a Flask-based REST API to serve dynamic recommendations for any user ID.

---


## 🧩 Folder Structure
movie-recommender-system/
│
├── recommender.py # Trains ALS model and saves it
├── app.py # Flask API for real-time recommendations
├── requirements.txt # Dependencies
├── model/ # Saved model files
├── data/ # Dataset files (or link)
├── screenshots/ # Project output images
└── report/ # Project report 




---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
bash - in Terminal
git clone https://github.com/rajkumarmath/movie-recommender-system.git
cd movie-recommender-system


**create a Virtual environment**
python3 -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows


**install dependencies**
pip install -r requirements.txt

**train the model**
python recommender.py

**run flask app**
python app.py

**test in browserr or postman**
http://127.0.0.1:5000/recommend?user=5


### sample output

Top recommendations for user 1:
  1. Duoluo tianshi (1995) – score=5.221
  2. Pather Panchali (1955) – score=5.058
  3. Wallace & Gromit (1996) – score=4.975
  4. Star Wars (1977) – score=4.967
  5. A Close Shave (1995) – score=4.880



