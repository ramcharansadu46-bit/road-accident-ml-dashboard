import os
import json
from flask import Flask, render_template, jsonify
from ml_engine import run_pipeline

app = Flask(__name__)

dashboard_data = None

def load_data_once():
    global dashboard_data
    if dashboard_data is None:
        with open("dashboard.json") as f:
            dashboard_data = json.load(f)

@app.route("/")
def index():
    load_data_once()
    return render_template(
        "index.html",
        data=json.dumps(dashboard_data),
        data_charts=dashboard_data["charts"]
    )

@app.route("/api/data")
def api_data():
    data = run_pipeline(data_path)   # reload every time
    return jsonify(data)