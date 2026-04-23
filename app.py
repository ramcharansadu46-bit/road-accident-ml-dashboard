import os
import json
from flask import Flask, render_template, jsonify
from ml_engine import run_pipeline

app = Flask(__name__)

data_path = os.path.join(os.path.dirname(__file__), "data.csv")

print("Loading data and training models...")
dashboard_data = run_pipeline(data_path)
print("Done! Starting server...")


@app.route("/")
def index():
    return render_template(
        "index.html",
        data=json.dumps(dashboard_data),
        data_charts=dashboard_data["charts"]
    )


@app.route("/api/data")
def api_data():
    return jsonify(dashboard_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)