import urllib

import cv2
import numpy as np
import roboflow
from flask import Flask, request
import openai
import json

openai.organization = "org-KGUkeR1rxambDb4iwt39rRQR"
openai.api_key = "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"

rf = roboflow.Roboflow(api_key='3ExsCVK8YEvuuYeNcGnS')

project = rf.workspace("nishanth").project("x-ray-mask")
model = project.version(1).model
colors = project.version(1).colors


def predict(image_url):
    prediction = model.predict(image_url)
    # prediction.plot()
    save_image(
        image=load_image(image_url),
        predictions=prediction.json()['predictions'],
        output_path='static/prediction.jpg'
    )
    prompt= "Give a brief diagnosis for this image based on json prediction data: "
    j = prediction.json()
    j['predictions'] = j['predictions'][:4]
    prompt += json.dumps(j)
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    print(response)
    print(response['choices'][0]['text'])
    res = response['choices'][0]['text']
    return res


def load_image(image_path):
    if "http://" in image_path:
        req = urllib.request.urlopen(image_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)  # 'Load it as it is'

        return image

    return cv2.imread(image_path)


def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))


def save_image(image, predictions, output_path):
    stroke_color = (255, 0, 0)
    for prediction in predictions:
        class_name = prediction["class"]
        if class_name in colors.keys():
            stroke_color = hex_to_rgb(colors[class_name][1:])
        points = [[int(p["x"]), int(p["y"])] for p in prediction["points"]]
        np_points = np.array(points, dtype=np.int32)
        cv2.fillPoly(
            image,
            [np_points],
            color=stroke_color
        )
    # Write image path
    cv2.imwrite(output_path, image)


# predict("./static/test.jpg")
app = Flask(__name__, static_url_path='',
            static_folder='static',)


@app.route("/")
def hello_world():
    return '''
    <html>
   <body>
      <form action = "/uploader" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
      </form>
   </body>
</html>
    '''


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    res = "prediction"
    if request.method == 'POST':
        f = request.files['file']
        f.save("./static/test.jpg")
        res = predict("./static/test.jpg")
        return f'''
        <html>
        <body>
          <div style='display:flex;gap:10px;'>
            <div>
              <p>Uploaded Image</p>
              <img src="/test.jpg" alt="test.jpg"/><br/>
            </div>
            <div>
              <p>Predicted Image</p>
              <img src="/prediction.jpg" alt="prediction.jpg"/>
              <p> {res} </p>
            </div>
          </div>
        </body>
        </html>
        '''
    return f'''
        <html>
        <body>
          <div style='display:flex;gap:10px;'>
            <div>
              <p>Uploaded Image</p>
              <img src="/test.jpg" alt="test.jpg"/><br/>
            </div>
            <div>
              <p>Predicted Image</p>
              <img src="/prediction.jpg" alt="prediction.jpg"/>
              <p> {res} </p>
            </div>
          </div>
        </body>
        </html>
        '''


app.run(host='0.0.0.0', port=6003)
