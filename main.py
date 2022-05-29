import os
from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def predict():
    if request.method == "POST":
        firebase_url = request.form["firebase_url"]
        print(firebase_url)
        res = firebase_url
        
        return render_template("index.html", result = res)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
