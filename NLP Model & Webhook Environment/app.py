print("🔧🔧🔧 STARTING APP.PY — NEW VERSION 🔧🔧🔧")

from flask import Flask
from routes import webhook_bp  # Import the webhook blueprint

app = Flask(__name__)
app.register_blueprint(webhook_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)