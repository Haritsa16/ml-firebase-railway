# ===========================
# Import Library
# ===========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load Data
# ===========================
generation_data = pd.read_csv(r"C:\Users\harit\OneDrive\Documents\Tugas Akhir\VR IoT PLTS\Machine Learning\Data\Plant_1_Generation_Data.csv")
weather_data = pd.read_csv(r"C:\Users\harit\OneDrive\Documents\Tugas Akhir\VR IoT PLTS\Machine Learning\Data\Plant_1_Weather_Sensor_Data.csv")

# ===========================
# Convert date-time format
# ===========================
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# ===========================
# Merge Data
# ===========================
df_solar = pd.merge(
    generation_data.drop(columns=['PLANT_ID','SOURCE_KEY','AC_POWER','DAILY_YIELD','TOTAL_YIELD']),
    weather_data.drop(columns=['PLANT_ID','SOURCE_KEY']),
    on='DATE_TIME'
)

# ===========================
# Buat target 4 jam ke depan
# ===========================
df_solar['DC_POWER_4h_ahead'] = df_solar['DC_POWER'].shift(-4)
df_solar = df_solar.dropna(subset=['DC_POWER_4h_ahead'])

# ===========================
# Buat lag features (opsional tapi disarankan)
# ===========================
df_solar['DC_POWER_t-1'] = df_solar['DC_POWER'].shift(1)
df_solar['DC_POWER_t-2'] = df_solar['DC_POWER'].shift(2)
df_solar['DC_POWER_t-3'] = df_solar['DC_POWER'].shift(3)

df_solar['IRRADIATION_t-1'] = df_solar['IRRADIATION'].shift(1)
df_solar['MODULE_TEMPERATURE_t-1'] = df_solar['MODULE_TEMPERATURE'].shift(1)

# Hapus baris dengan NaN karena lag
df_solar = df_solar.dropna()

# ===========================
# Tentukan fitur & target
# ===========================
feature_cols = ['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION',
                'DC_POWER_t-1','DC_POWER_t-2','DC_POWER_t-3',
                'IRRADIATION_t-1','MODULE_TEMPERATURE_t-1']

X = df_solar[feature_cols]
y = df_solar[['DC_POWER_4h_ahead']]

# ===========================
# Scaling
# ===========================
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# ===========================
# Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=21)

# ===========================
# Hyperparameter Tuning dengan GridSearchCV
# ===========================
params = {'n_neighbors': list(range(3, 51, 2))}
knn = neighbors.KNeighborsRegressor()
grid = GridSearchCV(knn, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train.ravel())

print("Best k:", grid.best_params_['n_neighbors'])

# ===========================
# Train final model
# ===========================
best_k = grid.best_params_['n_neighbors']
final_model = neighbors.KNeighborsRegressor(n_neighbors=best_k)
final_model.fit(X_train, y_train.ravel())

# ===========================
# Evaluasi Model
# ===========================
y_pred_scaled = final_model.predict(X_test).reshape(-1,1)
y_test_scaled = y_test

# De-scale hasil prediksi
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Error Metrics
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
r2 = r2_score(y_test_scaled, y_pred_scaled)

print(f"MSE (scaled): {mse:.6f}")
print(f"RMSE (scaled): {rmse:.6f}")
print(f"MAE (scaled): {mae:.6f}")
print(f"R² (scaled): {r2:.6f}")

# ===========================
# Plot Aktual vs Prediksi
# ===========================
plt.figure(figsize=(6,6))
plt.scatter(y_test_scaled, y_pred_scaled, alpha=0.5)
plt.xlabel("Aktual (scaled)")
plt.ylabel("Prediksi (scaled)")
plt.title("Aktual vs Prediksi KNN 4 jam ke depan")
plt.plot([0,1],[0,1],'r--')
plt.show()

# ===========================
# Simpan model & scaler
# ===========================
joblib.dump(final_model, "model_knn_4h_ahead.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# ===========================
# Prediksi Data Baru
# ===========================
# Contoh data baru: AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION,
# dan nilai lag sebelumnya DC_POWER & cuaca
data_baru = pd.DataFrame([[27, 40.3, 0.456, 5000, 4800, 4700, 0.42, 39.8]],
                        columns=feature_cols)

# Scaling
data_baru_scaled = scaler_X.transform(data_baru)

# Prediksi & de-scale
y_pred_baru_scaled = final_model.predict(data_baru_scaled).reshape(-1,1)
y_pred_baru = scaler_y.inverse_transform(y_pred_baru_scaled)
print("Hasil Prediksi DC_POWER 4 jam ke depan:", y_pred_baru)
