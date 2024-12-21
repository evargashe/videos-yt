from flask import Flask, url_for, render_template, request, redirect, session, url_for
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"

VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos')

@app.route("/")
def home():
    videos = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    return render_template("index.html", videos=videos)

@app.errorhandler(404)
def page_not_found(error):
    return (render_template('404.html'), 404)
