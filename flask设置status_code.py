from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    str_response = {'result': {
        "message": "成功",
        "data": {"videoPath": "aaaa",
                 "imagePath": "bbbb",
                 "jsonPath": "ccc"}
    }}
    return json.dumps(str_response, ensure_ascii=False), 404 #返回的第二个参数是设置status code

if __name__ == '__main__':
    app.run()
