from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Pastikan model berada di direktori yang sama dengan app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')

# Cek apakah model ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

# Muat model terlatih
model = joblib.load(MODEL_PATH)

# Memeriksa fitur yang diharapkan oleh model (jika tersedia)
try:
    expected_features = model.feature_names_in_
    print("Fitur yang diharapkan oleh model:", expected_features)
except AttributeError:
    print("Model tidak memiliki atribut 'feature_names_in_'. Pastikan feature_columns sesuai dengan yang digunakan saat pelatihan.")

# Definisikan nama-nama fitur sesuai dengan model
feature_columns = [
    'age',
    'trestbps',
    'chol',
    'fbs',
    'thalach',
    'exang',
    'oldpeak',
    'ca',
    'sex_Female',
    'sex_Male',
    'cp_atypical angina',
    'cp_non-anginal',
    'cp_typical angina',
    'restecg_lv hypertrophy',
    'restecg_st-t abnormality',
    'slope_downsloping',
    'slope_flat',
    'slope_upsloping',
    'thal_fixed defect',
    'thal_reversable defect'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None  # Inisialisasi variabel prediksi

    if request.method == 'POST':
        try:
            # Ekstrak dan preprocess data formulir
            age = int(request.form['Age'])
            trestbps = int(request.form['Resting_Blood_Pressure'])
            chol = int(request.form['Cholesterol_Measure'])
            fbs = 1 if request.form['Fasting_Blood_Sugar'] == 'True' else 0
            thalach = int(request.form['thalach'])
            exang = 1 if request.form['Exercise_induced_angina'] == 'True' else 0
            oldpeak = float(request.form['ST_Depression'])
            ca = int(request.form['blood'])  # blood vessels colored by fluoroscopy, 0-3

            # One-hot encode 'Sex'
            sex = request.form['Sex']
            sex_Female = 1 if sex == 'Female' else 0
            sex_Male = 1 if sex == 'Male' else 0

            # One-hot encode 'Chest_Pain_Type'
            cp = request.form['Chest_Pain_Type']
            cp_atypical_angina = 1 if cp == 'Atypical Angina' else 0
            cp_non_anginal = 1 if cp == 'Non-Anginal' else 0
            cp_typical_angina = 1 if cp == 'Typical Angina' else 0

            # One-hot encode 'ECG_at_resting'
            restecg = request.form['ECG_at_resting']
            restecg_lv_hypertrophy = 1 if restecg == 'LV Hypertrophy' else 0
            restecg_st_t_abnormality = 1 if restecg == 'ST-T Abnormality' else 0

            # One-hot encode 'Slope'
            slope = request.form['Slope']
            slope_downsloping = 1 if slope == 'downsloping' else 0
            slope_flat = 1 if slope == 'flat' else 0
            slope_upsloping = 1 if slope == 'upsloping' else 0

            # One-hot encode 'Thalassemia'
            thal = request.form['Thalassemia']
            thal_fixed_defect = 1 if thal == 'fixed' else 0
            thal_reversable_defect = 1 if thal == 'reversable' else 0

            # Buat dictionary data input dengan one-hot encoding
            input_data_dict = {
                'age': age,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'ca': ca,
                'sex_Female': sex_Female,
                'sex_Male': sex_Male,
                'cp_atypical angina': cp_atypical_angina,
                'cp_non-anginal': cp_non_anginal,
                'cp_typical angina': cp_typical_angina,
                'restecg_lv hypertrophy': restecg_lv_hypertrophy,
                'restecg_st-t abnormality': restecg_st_t_abnormality,
                'slope_downsloping': slope_downsloping,
                'slope_flat': slope_flat,
                'slope_upsloping': slope_upsloping,
                'thal_fixed defect': thal_fixed_defect,
                'thal_reversable defect': thal_reversable_defect
            }

            # Konversi ke DataFrame
            input_data = pd.DataFrame([input_data_dict])

            # Pastikan semua fitur ada
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Urutkan kolom sesuai dengan feature_columns
            input_data = input_data[feature_columns]

            # Cetak untuk debugging
            print("Jumlah fitur yang dikirimkan:", input_data.shape[1])
            print("Fitur yang dikirimkan:", input_data.columns.tolist())

            # Pastikan jumlah fitur sesuai
            if input_data.shape[1] != len(feature_columns):
                raise ValueError(f"Jumlah fitur tidak sesuai. Diharapkan {len(feature_columns)}, tetapi mendapatkan {input_data.shape[1]}.")

            # Prediksi
            prediction = model.predict(input_data)[0]
            prediction_text = 'Positive' if prediction == 1 else 'Negative'

        except Exception as e:
            # Menangani semua exception dan menampilkan pesan error
            prediction_text = f"An error occurred: {str(e)}"
            print(f"Error during prediction: {str(e)}")

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    # Jalankan Flask di host dan port yang sesuai
    app.run(debug=True)
