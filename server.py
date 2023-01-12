#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from gevent import pywsgi
from flask import Flask, request, jsonify
from lightner_flask.flask_predict import *

base_path = os.path.dirname(os.path.dirname(__file__))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# 首先加载模型，服务一起来模型也要跟着起来后续不用反复加载，浪费时间
trainer, model2 = get_model_first_upload()
logger.info("Load model successful!")

app = Flask("lightner-app")


@app.route("/ner_server", methods=['POST'])
def ner_server():
    bio_list = list()
    data = request.json
    content_list = data['content_list']
    try:
        for content in content_list:
            string_list = [i + "\t" + "O\n" for i in [j for j in content.strip("\t")]]
            string_list.append("\n")
            bio_list += string_list
        texts, labels = ner_server_main(bio_list, trainer, model2)
        data_format = {"content_list": content_list, "text": texts, "labels": labels}
        bio_label_list = get_predict_label(data_format)
        output_result = get_words_bio_label(bio_label_list)
        return jsonify(output_result)
    except Exception as e:
        return jsonify(e)


if __name__ == "__main__":
    logger.info("启动light-few-shot-ner预测服务")
    server = pywsgi.WSGIServer(('0.0.0.0', 12345), app)
    server.serve_forever()
    # app.run(host='0.0.0.0', port=12345, threaded=True, debug=True)
