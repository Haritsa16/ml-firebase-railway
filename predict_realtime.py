import pyrebase
import pandas as pd
import joblib
from collections import deque
import time

# ==========================
# 1. Load model & scaler
# ==========================
model = joblib.load("model_knn_4h_ahead.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ==========================
# 2. Firebase Config (pakai Pyrebase, bukan firebase_admin)
# ==========================
firebase_config = {
  "apiKey": "AIzaSyBFjzL-7TSHCdEp24-9N1taB7g5QrfzGNo",
  "authDomain": "coba-esp-4.firebaseapp.com",
  "databaseURL": "https://coba-esp-4-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "coba-esp-4",
  "storageBucket": "coba-esp-4.firebasestorage.app",
  "messagingSenderId": "823614837927",
  "appId": "1:823614837927:web:1d14d80a806c8ad7697782",
  "measurementId": "G-R9VDE2FNGY"
};

firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

# ==========================
# 3. Setup history buffers
# ==========================
dc_power_history = deque([0, 0, 0], maxlen=3)
irradiance_history = deque([0, 0, 0], maxlen=3)
module_temp_history = deque([0, 0, 0], maxlen=3)

# ==========================
# 4. Realtime loop
# ==========================
while True:
    try:
        data = db.child("devices/esp32_1/sensor").get().val()

        if not data:
            print("Data kosong, tunggu...")
            time.sleep(5)
            continue

        # 5. Mapping data ke fitur model
        data_mapped = {
            'AMBIENT_TEMPERATURE': data.get('temp_dht', 0),
            'MODULE_TEMPERATURE': data.get('temp_ds18', 0),
            'IRRADIATION': data.get('irradiance', 0),
            'DC_POWER_t-1': dc_power_history[-1],
            'DC_POWER_t-2': dc_power_history[-2],
            'DC_POWER_t-3': dc_power_history[-3],
            'IRRADIATION_t-1': irradiance_history[-1],
            'MODULE_TEMPERATURE_t-1': module_temp_history[-1]
        }

        print("Data realtime:", data)
        print("Fitur untuk model:", data_mapped)

        data_baru = pd.DataFrame([data_mapped])

        # 6. Scaling & prediksi
        data_scaled = scaler_X.transform(data_baru)
        y_pred_scaled = model.predict(data_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        hasil_prediksi = float(y_pred[0][0])/10

        print("Prediksi DC_POWER 4 jam ke depan:", hasil_prediksi)

        # 7. Simpan hasil prediksi ke Firebase
        tanggal = time.strftime("%Y-%m-%d")
        jam = time.strftime("%H:%M:%S")

        db.child("devices/esp32_1/prediksi").set({
            "dc_power_predicted": hasil_prediksi,
            "tanggal": tanggal,
            "jam": jam
        })

        db.child("devices/esp32_1/sensor").update({
            "prediksi": hasil_prediksi
        })

        # Update ke log terakhir (kalau ada)
        logs = db.child(f"devices/esp32_1/sensorLog/{tanggal}").get().val()
        if logs:
            last_time = list(logs.keys())[-1]
            db.child(f"devices/esp32_1/sensorLog/{tanggal}/{last_time}").update({
                "prediksi": hasil_prediksi
            })
            print(f"Prediksi ditambahkan ke {tanggal}/{last_time}")
        else:
            print("Belum ada log sensor dari ESP hari ini.")

        # 8. Update history
        dc_power_history.append(data.get('dc_power', 0))
        irradiance_history.append(data.get('irradiance', 0))
        module_temp_history.append(data.get('temp_ds18', 0))

        time.sleep(5)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)
