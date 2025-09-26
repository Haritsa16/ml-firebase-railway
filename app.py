from flask import Flask, render_template, jsonify
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib

# ==============================
# 1. Setup Flask
# ==============================
app = Flask(__name__)

# ==============================
# 2. Load Model & Scaler
# ==============================
model = joblib.load("model_knn_4h_ahead.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ==============================
# 3. Setup Firebase
# ==============================
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://coba-esp-3-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ==============================
# 4. Route untuk Prediksi
# ==============================
@app.route("/")
def index():
    ref = db.reference("devices/esp32_1/sensor")
    data = ref.get()

    feature_cols = ["irradiance", "temp_dht", "temp_ds18", "humidity", "lux"]

    df = pd.DataFrame([[data.get(col, 0) for col in feature_cols]], columns=feature_cols)

    df_scaled = scaler_X.transform(df)
    y_pred_scaled = model.predict(df_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return jsonify({
        "sensor_data": data,
        "predicted_dc_power": float(y_pred[0][0])
    })

# ==============================
# 5. Run Flask
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
