from flask import Flask, request, jsonify
import joblib
import pandas as pd

# untuk initialize app
app = Flask(__name__)

# ambil codes yg disimpan dalam pickle
model = joblib.load('lr.pkl')
features = joblib.load('features.pkl')
std = joblib.load('std.pkl')
transformation = joblib.load('transform.pkl')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # untuk terima input json (raw input di postman)
        data = request.get_json()

        # ubah json jadi dataframe karena model bisa terima data dalam bentuk dataframe
        df = pd.DataFrame([data])

        # preprocess input json dengan cara yang sama

        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

        df = df.reindex(columns=transformation.columns, fill_value=0)

        df = df[features]

        df = std.transform(df)

        # predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0].max()

        return jsonify({"prediction": int(pred), "confidence": float(prob)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)