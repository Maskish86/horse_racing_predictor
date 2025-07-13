from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return "Service is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # For example, echo back
    return jsonify({"input": data})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
