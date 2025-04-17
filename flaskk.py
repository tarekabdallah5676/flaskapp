from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

app = Flask(__name__)
ort_session = ort.InferenceSession("last.onnx")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    files = request.files.getlist('imagefile')
    os.makedirs('static/images', exist_ok=True)

    predictions = []

    for file in files:
        image_path = os.path.join('static', 'images', file.filename)
        file.save(image_path)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        inputs = {ort_session.get_inputs()[0].name: img_array}
        outputs = ort_session.run(None, inputs)

        prediction = outputs[0]
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])

        label = "Homemade" if class_id == 0 else "Machinemade"
        result = f"{label} ({confidence * 100:.2f}%)"

        predictions.append({
            'filename': file.filename,
            'result': result,
            'path': image_path
        })

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)





