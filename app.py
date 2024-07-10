from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import joblib
import numpy as np
import json
import os
import locale

app = Flask(__name__)

# Path ke model yang telah dilatih
MODEL_PATH = 'lr_model_HousePrice.pkl'
model = joblib.load(MODEL_PATH)

# Path ke file histori prediksi
HISTORY_FILE = 'prediction_history.json'

# Mapping merek dari string ke nilai numerik
LOCATION_DICT = {'Jakarta': 0, 'Depok': 1, 'Tangerang': 2}

# Memastikan file histori ada, jika tidak, buat file baru
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# Fungsi untuk memvalidasi data permintaan
def validate_request_data(data):
    required_fields = ["Location", "Land Size", "Building Size", "Num Rooms", "Num Floors"]
    for field in required_fields:
        if field not in data:
            return False, f"{field} is required"
        if field == "Location" and data[field] not in LOCATION_DICT:
            return False, "Invalid Location"
        if field in ["Land Size", "Building Size", "Num Rooms", "Num Floors"]:
            try:
                float(data[field])
            except ValueError:
                return False, f"Invalid type for {field}"
    return True, ""

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    # Menangani kesalahan permintaan yang buruk
    return jsonify(error=str(e)), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validasi data permintaan
    is_valid, error_message = validate_request_data(data)
    if not is_valid:
        return jsonify({'error': error_message}), 400

    # Mendapatkan fitur dari data
    location_str = data['Location']
    location = LOCATION_DICT[location_str]
    land_size = float(data['Land Size'])
    building_size = float(data['Building Size'])
    num_rooms = float(data['Num Rooms'])
    num_floors = float(data['Num Floors'])
    ratio = land_size/(building_size+1)

    # Membuat array fitur
    features = np.array([[location, land_size, building_size, num_rooms, num_floors, ratio]])

    # Melakukan prediksi
    try:
        prediction = model.predict(features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Mengatur locale ke Indonesia untuk format Rupiah
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
    prediction_formatted = locale.currency(prediction[0]*160, grouping=True)

    prediction_result = {'Location': location_str, 'Land Size': land_size, 'Building Size': building_size, 'Num Rooms': num_rooms, 'Num Floors': num_floors, 'Ratio': ratio, 'Prediction': prediction_formatted}
    
    # Menyimpan hasil prediksi ke file histori
    with open(HISTORY_FILE, 'r+') as f:
        history = json.load(f)
        history.append(prediction_result)
        f.seek(0)
        json.dump(history, f, indent=4)

    # Mengembalikan hasil prediksi sebagai JSON
    return jsonify({'prediction': prediction_formatted})

@app.route('/history', methods=['GET'])
def get_history():
    # Mendapatkan histori prediksi dari file
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    return jsonify(history)

if __name__ == '__main__':
    # Menjalankan aplikasi Flask
    app.run(debug=True)
