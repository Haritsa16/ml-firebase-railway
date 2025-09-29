import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib
from collections import deque
import time
import os, json  # Tambahan untuk baca env

# ==========================
# 1. Load model & scaler
# ==========================
model = joblib.load("model_knn_4h_ahead.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ==========================
# 2. Setup Firebase dari Environment Variable
# ==========================
firebase_config = json.loads(os.environ["FIREBASE_CREDENTIALS"])  # Ambil dari Railway
cred = credentials.Certificate(firebase_config)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://coba-esp-3-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ==========================
# 3. Setup history buffers (lag features)
# ==========================
dc_power_history = deque([0, 0, 0], maxlen=3)
irradiance_history = deque([0], maxlen=1)
module_temp_history = deque([0], maxlen=1)

# ==========================
# 4. Realtime loop
# ==========================
while True:
    try:
        ref = db.reference("devices/esp32_1/sensor")
        data = ref.get()

        if not data:
            print("Data kosong, tunggu...")
            time.sleep(5)
            continue

        # ==========================
        # 5. Mapping Firebase -> model feature
        # ==========================
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

        data_baru = pd.DataFrame([data_mapped])

        # ==========================
        # 6. Scaling & prediksi
        # ==========================
        data_scaled = scaler_X.transform(data_baru)
        y_pred_scaled = model.predict(data_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        hasil_prediksi = float(y_pred[0][0])

        print("Data realtime:", data)
        print("Prediksi DC_POWER 4 jam ke depan:", hasil_prediksi)

        # ==========================
        # 7a. Simpan hasil prediksi ke devices/esp32_1/prediksi
        # ==========================
        pred_ref = db.reference("devices/esp32_1/prediksi")
        pred_ref.set({
            "dc_power_predicted": hasil_prediksi,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # ==========================
        # 7b. Update field prediksi di devices/esp32_1/sensor
        # ==========================
        sensor_ref = db.reference("devices/esp32_1/sensor")
        sensor_ref.update({
            "prediksi": hasil_prediksi
        })

        # ==========================
        # 7c. Simpan ke sensorLog/{tanggal}/{jam}
        # ==========================
        tanggal = time.strftime("%Y-%m-%d")
        jam = time.strftime("%I:%M:%S %p")  # contoh: 02:45:39 PM

        log_ref = db.reference(f"devices/esp32_1/sensorLog/{tanggal}/{jam}")
        log_ref.set({
            "humidity": data.get("humidity", 0),
            "irradiance": data.get("irradiance", 0),
            "lux": data.get("lux", 0),
            "temp_dht": data.get("temp_dht", 0),
            "temp_ds18": data.get("temp_ds18", 0),
            "prediksi": hasil_prediksi
        })

        # ==========================
        # 8. Update history buffers
        # ==========================
        dc_power_history.append(data.get('dc_power', 0))
        irradiance_history.append(data.get('irradiance', 0))
        module_temp_history.append(data.get('temp_ds18', 0))

        time.sleep(5)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)
