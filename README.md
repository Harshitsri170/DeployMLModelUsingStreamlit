# DeployMLModelUsingStreamlit
This project demonstrates how to train multiple regression models and deploy them using a Streamlit web app. Users can select an ML algorithm from the sidebar, input data interactively through sliders, and view real-time predictions with model accuracy and optional feature importance.

🚀 Features
✅ Train 4 popular ML models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regressor (SVR)

✅ Deploy using Streamlit with a clean and intuitive UI

✅ Interactive sliders for feature input

✅ Model selection via sidebar

✅ Real-time prediction of California housing prices

✅ R² Score (accuracy) shown for selected model

✅ Feature importance visualization (for tree-based models)

🧠 Dataset Used
This app uses the California Housing dataset from sklearn.datasets, which includes real-world housing data such as:

Average number of rooms

Latitude, longitude

Median income

Population, and more...

📂 Project Structure
bash
Copy
Edit
ml_model_deploy_app/
│
├── app.py                 # Streamlit UI
├── train_models.py        # Trains and saves all models
├── model_utils.py         # Loads saved models
├── requirements.txt       # Dependencies list
├── README.txt             # Setup guide (for Windows)
└── models/                # (Auto-created) Contains trained model .pkl files
💻 How to Run Locally (Windows)
Clone the repo or download the ZIP and extract it

Open the folder in VS Code

Create a virtual environment:

bash
Copy
Edit
python -m venv venv
Activate the environment:

bash
Copy
Edit
.\venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train the models:

bash
Copy
Edit
python train_models.py
Run the app:

bash
Copy
Edit
streamlit run app.py


📌 Future Enhancements
Add classification model support (e.g., Iris or Breast Cancer)

Add input validation

Deploy to Streamlit Cloud or HuggingFace Spaces

Add model explanation using SHAP

🙌 Acknowledgements
Streamlit

Scikit-learn

California Housing Dataset
