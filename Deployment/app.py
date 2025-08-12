from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os, json

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/best.pt')
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    print('⚠️ best.pt not found at', MODEL_PATH, '- upload your trained weights.')

REC_PATH = os.path.join(os.path.dirname(__file__), '../../utils/recommendations.json')
with open(REC_PATH, 'r') as f:
    RECS = json.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result=None, error='No file uploaded')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result=None, error='Empty filename')

    img_path = os.path.join(os.path.dirname(__file__), 'static', file.filename)
    file.save(img_path)

    if model is None:
        return render_template('index.html', result=None, error='Model not loaded. Add models/best.pt')

    results = model.predict(img_path)
    soil = None
    conf = None
    for r in results:
        soil = r.names[r.probs.top1]
        conf = float(r.probs.top1conf)

    rec = RECS.get(soil, {"crops": [], "tips": "No tips available."})

    return render_template('index.html', result={
        "soil": soil, "confidence": f"{conf:.2f}", "crops": rec["crops"], "tips": rec["tips"], "image": f"static/{file.filename}"
    }, error=None)

if __name__ == '__main__':
    app.run(debug=True)
