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
        if request.method == 'POST':
            cover_image = request.form.get('cover')
            secret_image = request.form.get('secret')
            if cover_image[:4]=='http':
                print(cover_image[:4])
                print("getting url image ... ")
                urllib.request.urlretrieve(cover_image, IMAGE_PATH_DEMO_API + 'cover.png')
                urllib.request.urlretrieve(secret_image, IMAGE_PATH_DEMO_API + 'secret.png')
                steg_image = encode(net, optim, IMAGE_PATH_DEMO_API + 'cover.png', IMAGE_PATH_DEMO_API + 'secret.png')
            else:
                print("getting path image ... ")
                steg_image = encode(net, optim, cover_image, secret_image)
                print(steg_image)
            # cover_image.save(IMAGE_PATH_DEMO_API + 'cover.png')
            # secret_image.save(IMAGE_PATH_DEMO_API + 'secret.png')
            # encode(net, optim, IMAGE_PATH_DEMO_API + 'cover.png', IMAGE_PATH_DEMO_API + 'secret.png')
            return {"response": f"steg image path {steg_image}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)