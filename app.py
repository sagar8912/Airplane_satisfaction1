from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ===============================
# BASE DIRECTORY
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# MODEL PATH (MAKE SURE FILE EXISTS)
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "xgb_classifier.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

# ===============================
# LOAD MODEL
# ===============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ---------- USER INPUT ----------
    age = int(request.form["age"])
    flight_distance = int(request.form["flight_distance"])
    dep_delay = int(request.form["dep_delay"])
    arr_delay = int(request.form["arr_delay"])

    comfort = int(request.form["comfort"])
    service = int(request.form["service"])
    digital = int(request.form["digital"])
    airport = int(request.form["airport"])
    clean = int(request.form["clean"])

    # ---------- MODEL INPUT ----------
    data = {
        "Age": age,
        "Flight Distance": flight_distance,
        "Seat comfort": comfort,
        "Departure/Arrival time convenient": airport,
        "Food and drink": service,
        "Gate location": airport,
        "Inflight wifi service": digital,
        "Inflight entertainment": digital,
        "Online support": digital,
        "Ease of Online booking": digital,
        "On-board service": service,
        "Leg room service": comfort,
        "Baggage handling": service,
        "Checkin service": service,
        "Cleanliness": clean,
        "Online boarding": digital,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay,

        # ----- Encoded categorical values -----
        "Gender": 0,               # Female
        "Customer Type": 1,        # Loyal Customer
        "Type of Travel": 1,       # Business travel
        "Class_Business": 0,
        "Class_Eco": 1,
        "Class_Eco Plus": 0
    }

    df = pd.DataFrame([data])

    # ---------- ALIGN FEATURES ----------
    expected_cols = model.get_booster().feature_names
    df = df.reindex(columns=expected_cols, fill_value=0)

    # ---------- PREDICTION ----------
    prob = model.predict_proba(df)[0][1]
    confidence = round(prob * 100, 2)

    avg_rating = (comfort + service + digital + airport + clean) / 5

    if avg_rating >= 3.5 and prob >= 0.6:
        result = "Satisfied 😊"
    else:
        result = "Dissatisfied 😞"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence
    )


# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)












