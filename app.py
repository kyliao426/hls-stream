from flask import render_template, Flask, send_from_directory
import subprocess
import time
app = Flask(__name__)


@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video/<string:file_name>')
def stream(file_name):
    video_dir = './video'
    return send_from_directory(directory=video_dir, path=file_name)


if __name__ == '__main__':
    subprocess.Popen('ffmpeg -y -f vfwcap -r 25 -rtbufsize 1024M  -i 0 -f hls -hls_time 2 ./video/live.m3u8', shell=True)
    time.sleep(10)
    subprocess.Popen('python track.py --save-vid --output video')
    time.sleep(5)
    app.run()
