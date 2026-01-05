import random
import time

import requests

url = "http://localhost:8000/predict"


# 🔴 Генерируем клиента, который ГАРАНТИРОВАННО должен уйти
# (Максимально плохие условия для удержания)
def generate_toxic_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 1,  # Пенсионеры чаще уходят
        "Partner": "No",  # Нет привязки к семье
        "Dependents": "No",
        "tenure": 1,  # Только пришел (1 месяц)
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",  # Самый дорогой и проблемный инет
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",  # Никакой поддержки
        "StreamingTV": "Yes",  # Набрал услуг...
        "StreamingMovies": "Yes",  # ...чтобы чек был огромным
        "Contract": "Month-to-month",  # Никаких обязательств
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",  # Самый "ненадежный" метод
        "MonthlyCharges": 118.75,  # МАКСИМАЛЬНО возможная цена в датасете
        "TotalCharges": 118.75,  # Равно месячной, так как 1й месяц
    }


# 🟢 Идеальный клиент (Лояльный)
def generate_loyal_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,  # С нами 6 лет (максимум)
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",  # Нет интернета - нет проблем
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",  # Контракт на 2 года
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,  # Минимальная цена
        "TotalCharges": 1400.0,
    }


print("🚀 Starting EXTREME Load Test...")

while True:
    # Чередуем: Плохой -> Хороший
    if random.random() > 0.5:
        data = generate_toxic_customer()
        type_cust = "🤬 TOXIC"
    else:
        data = generate_loyal_customer()
        type_cust = "😇 LOYAL"

    try:
        response = requests.post(url, json=data)

        if response.status_code == 200:
            res = response.json()
            pred = res["message"]
            prob = res["probability"]

            icon = "🔴" if prob > 0.5 else "🟢"
            print(f"Sent: {type_cust} | Model: {prob:.4f} -> {icon} {pred}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"🚨 Connection error: {e}")

    time.sleep(0.2)  # Шлем быстро, чтобы забить графики
