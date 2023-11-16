from flask import Flask, request
import urllib.request
# from PIL import Image
# import requests
from demo_app import encode, load_model
from config import *
app = Flask(__name__)
net, optim = load_model()

@app.route("/encode_image", methods=["POST", "GET"])
def main():
    try:
        # if request.method == 'POST':
        #     # print("request",request.form)
        #     data = request.json
        #     print(data)
        #     cover_url = data.get("cover")
        #     urllib.request.urlretrieve(cover_url, IMAGE_PATH_DEMO_API + 'cover.png')
        #
        #     secret_url = data.get("secret")
        #     urllib.request.urlretrieve(cover_url, IMAGE_PATH_DEMO_API + 'secret.png')
        #
        #     # cover_image = Image.open(IMAGE_PATH_DEMO_API + 'input.png')
        #     encode(net, optim, IMAGE_PATH_DEMO_API + 'cover.png', IMAGE_PATH_DEMO_API + 'secret.png')
        #     # image_name = image.filename
        #     # print(image)
        #     # image = Image.open(requests.get(url, stream=True).raw)
        #
        #     # if '.jpg' in image_name:
        #     #
        #     #     image.save(image_name)
        #     #
        #     #     return {"response": "file saved successfully in your current durectory"}
        #     # elif '.jpeg' in image_name:
        #     #     image.save(image_name)
        #     #
        #     #     return {"response": "file saved successfully in your current durectory"}
        #     # else:
        #     #     return {"error": "select you image file"}
        #     return {"response": "file saved successfully in your current durectory"}
        if request.method == 'POST':
            # print("request",request.form)
            cover_image = request.files['cover']
            secret_image = request.files['secret']
            cover_image.save(IMAGE_PATH_DEMO_API + 'cover.png')
            secret_image.save(IMAGE_PATH_DEMO_API + 'secret.png')
            encode(net, optim, IMAGE_PATH_DEMO_API + 'cover.png', IMAGE_PATH_DEMO_API + 'secret.png')
            return {"response": "file saved successfully in your current durectory"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run("0.0.0.0", debug=False)