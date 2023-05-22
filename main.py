import os.path

from flask.views import MethodView
from wtforms import Form, StringField, SubmitField
from flask import Flask, render_template, request

import webbrowser

from image_gen import generate_image

app = Flask(__name__)


class HomePage(MethodView):

    def get(self):
        return render_template('index.html')

class ImageResult(MethodView):

    def get(self):
        generate_image()
        return render_template('image_result.html')


app.add_url_rule('/', view_func=HomePage.as_view('home_page'))
app.add_url_rule('/image_result', view_func=ImageResult.as_view('image_result'))

app.run(host='0.0.0.0', debug=True, port=8080)
