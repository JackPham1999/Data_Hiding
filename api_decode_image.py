from flask import Flask, request
import urllib.request
# from PIL import Image
# import requests
from demo_app import decode, load_model
from config import *
app = Flask(__name__)
net, optim = load_model()


@app.route("/decode_image", methods=["POST", "GET"])
def main():
    try:
        print("hello")
        if request.method == 'POST':
            # print("request",request.form)
            image = request.files['steg']
            image.save(IMAGE_PATH_DEMO_API + 'steg.png')
            recover_image, recover_cover = decode(net, optim, IMAGE_PATH_DEMO_API + 'steg.png')
            return {"response": f"recover image path {recover_image} and {recover_cover}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run("0.0.0.0", port=5001,debug=True)