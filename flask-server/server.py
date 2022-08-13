import base64
from cProfile import run
from flask import Flask,request,jsonify
import io
from io import BytesIO
from keras.models import load_model
from PIL import Image,ImageChops,ImageOps
import base64
import numpy as np

app=Flask(__name__)

def get_model():
    global model
    model=load_model("mnist.h5")
    print("model loaded")

print("loading model")
get_model()


def remove_alpha_channel(image):
  if image.mode in ["RGBA", "P"]:
    image = image.convert("RGB")
  return image

def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image

def pad_image(image):
    return ImageOps.expand(image, border=30, fill='#fff')


def resize_image(image):
    return image.resize((28, 28), Image.LINEAR)


def to_grayscale(image):
    return image.convert('L')

def invert_colors(image):
    return ImageOps.invert(image)

def expand_dims(image):
    image = np.expand_dims(image,-1) 
    image=image[None]
    print(image.shape)

    return image



@app.route('/predict',methods=['POST'])
def predict():
  image_data = request.json
  preprocessImage=image_data["image"]
  strImage =preprocessImage.split("base64,")[1];
  bytes_encoded = bytes(strImage, encoding="ascii")
  im = Image.open(BytesIO(base64.b64decode(bytes_encoded)))
  im=remove_alpha_channel(im)
  im=trim_borders(im)
  im=pad_image(im)
  im=resize_image(im)
  im=to_grayscale(im)
  im=invert_colors(im)
  test_image=expand_dims(im)
  predication=model.predict(test_image)
  result=(np.argmax(predication))
  im.save("image.jpeg")
  print(type(result))
  return {
        "result":int(result),
    }

if __name__=='__main__':
    app.run(port=5000,debug=True)