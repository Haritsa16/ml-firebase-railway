import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib
from collections import deque
import time
import os, json

# ==========================
# 1. Load model & scaler
# ==========================
model = joblib.load("model_knn_4h_ahead.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ==========================
# 2. Setup Firebase
# ==========================
firebase_json = os.environ.get("FIREBASE_CREDENTIALS", None)
if not firebase_json:
    raise RuntimeError("❌ FIREBASE_CREDENTIALS tidak ditemukan di environment!")

try:
    firebase_config = json.loads(firebase_json)
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://coba-esp-4-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    print("✅ Firebase terhubung!")
except Exception as e:
    raise RuntimeError(f"❌ Gagal inisialisasi Firebase: {e}")

# ==========================
# 3. History buffers
# ==========================
dc_power_history = deque([0, 0, 0], maxlen=3)
irradiance_history = deque([0, 0, 0], maxlen=3)
module_temp_history = deque([0, 0, 0], maxlen=3)

# ==========================
# 4. Loop realtime
# ==========================
while True:
    try:
        ref = db.reference("devices/esp32_1/sensor")
        data = ref.get()

        if not data:
            print("⚠️ Data kosong, tunggu 5 detik...")
            time.sleep(5)
            continue

        print("\n🔥 Data realtime:", data)

        # ==========================
        # 5. Mapping data
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

        print("🧠 Fitur ke model:", data_mapped)

        # ==========================
        # 6. Scaling & prediksi
        # ==========================
        df = pd.DataFrame([data_mapped])
        data_scaled = scaler_X.transform(df)
        y_pred_scaled = model.predict(data_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        hasil_prediksi = float(y_pred[0][0])

        print(f"✅ Prediksi DC_POWER 4 jam ke depan: {hasil_prediksi:.2f}")

        # ==========================
        # 7. Simpan hasil prediksi
        # ==========================
        tanggal = time.strftime("%Y-%m-%d")
        jam = time.strftime("%H:%M:%S")

        # --- a. Simpan realtime prediksi
        try:
            pred_ref = db.reference("devices/esp32_1/prediksi")
            pred_ref.set({
                "dc_power_predicted": hasil_prediksi,
                "tanggal": tanggal,
                "jam": jam
            })
            print("📡 Prediksi disimpan ke devices/esp32_1/prediksi ✅")
        except Exception as e:
            print("❌ Gagal menyimpan ke prediksi:", e)

        # --- b. Update field di sensor utama
        try:
            sensor_ref = db.reference("devices/esp32_1/sensor")
            sensor_ref.update({"prediksi": hasil_prediksi})
            print("📡 Field 'prediksi' di sensor diperbarui ✅")
        except Exception as e:
            print("❌ Gagal update field di sensor:", e)

        # --- c. Update di log terakhir (sensorLog)
        try:
            log_ref = db.reference(f"devices/esp32_1/sensorLog/{tanggal}")
            last_logs = log_ref.order_by_key().limit_to_last(1).get()

            if last_logs and isinstance(last_logs, dict):
                last_time = list(last_logs.keys())[0]
                # Normalisasi format jam (hindari "1:08:40 PM")
                try:
                    if "M" in last_time:
                        parsed_time = time.strptime(last_time, "%I:%M:%S %p")
                        last_time = time.strftime("%H:%M:%S", parsed_time)
                except:
                    pass

                log_ref.child(last_time).update({"prediksi": hasil_prediksi})
                print(f"📡 Prediksi ditambahkan ke {tanggal}/{last_time} ✅")
            else:
                print("⚠️ Tidak ada log valid untuk hari ini.")
        except Exception as e:
            print("❌ Gagal update ke log terakhir:", e)

        # ==========================
        # 8. Update history
        # ==========================
        dc_power_history.append(data.get('dc_power', 0))
        irradiance_history.append(data.get('irradiance', 0))
        module_temp_history.append(data.get('temp_ds18', 0))

        time.sleep(5)

    except Exception as e:
        print("🔥 Error utama:", e)
        time.sleep(5)
