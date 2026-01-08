from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# -------- LOAD MODEL --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_cassifer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

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

    # ---------- NUMERIC DATA ONLY ----------
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

        # ---------- DUMMY VARIABLES (NUMERIC) ----------
        "Gender": 0,
        "Customer Type": 1,
        "Type of Travel": 1,
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

    if avg_rating <= 2.5 or prob < 0.55:
        result = "Dissatisfied 😞"
    elif avg_rating >= 3.5 and prob >= 0.6:
        result = "Satisfied 😊"
    else:
        result = "Dissatisfied 😞"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)










