import ntpath
import os
import urllib.request
import random
import torch
import request
import traceback
from Utils.ad_generator import AdGenerator
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import cross_origin

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ad_generator = AdGenerator()
uploaded_advertisements = dict()


@app.route('/api/')
def hello_world():
    return jsonify(error=False, message="stub_page")


@cross_origin(origins="*")
@app.route('/api/upload/', methods=['OPTIONS'])
def upload_file_opts():
    return jsonify(error=False, message="dummy_options")


@cross_origin(origins="*")
@app.route('/api/img/', methods=['GET'])
def api_adv_get_img():
    file_id = request.args['id']
    file_ext = uploaded_advertisements[file_id]['_imgExt']
    return send_file("uploads/%s%s" % (file_id, file_ext), mimetype='image/jpg')


@cross_origin(origins="*")
@app.route('/api/list/', methods=['GET'])
def api_adv_list():
    entry_list = []
    for k, v in uploaded_advertisements.items():
        entry_list.append(v)

    return jsonify(error=False, body=entry_list)


@cross_origin(origins="*")
@app.route('/api/place/', methods=['POST'])
def api_adv_place():
    try:
        r_data = request.json
        if '_id' not in r_data:
            return jsonify(error=True, message="No image id field")

        for k, d_entry in uploaded_advertisements.items():
            if d_entry['_id'] == r_data['_id']:
                r_data['_imgExt'] = d_entry['_imgExt']
                uploaded_advertisements[k] = r_data
                return jsonify(error=False, body="ok")

        return jsonify(error=True, message="That image is not exists, server restarted?")
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=True, message=str(e))


@cross_origin(origins="*")
@app.route('/api/upload/', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']

        file_id = str(hex(random.getrandbits(128)))
        upload_file_name, upload_file_extension = os.path.splitext(file.filename)
        upload_file_name = os.path.join(app.config['UPLOAD_FOLDER'], '%s%s' % (file_id, upload_file_extension))
        disk_file = open(upload_file_name, "bw")
        disk_file.write(file.read())
        print("File '%s' are write to disk" % upload_file_name)
        uploaded_advertisements[file_id] = {
            "_id": file_id,
            "_imgExt": upload_file_extension,
        }
        result = ad_generator.get_ad_json(upload_file_name, file_id)
        # Here file is
        # file.save(f)
        return jsonify(error=False, body=result)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=True, message=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8756)
