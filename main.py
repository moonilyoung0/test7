import os
from flask import Flask, render_template, request
import test_model

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def predict():
    if request.method == "POST":
        firebase_url = request.form["firebase_url"]
        print(firebase_url)

        video = test_model.download_video(firebase_url)
        print(video)
        
        res = test_model.predict_video(video)
        print(res)
        
        return render_template("index.html", result = res)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
