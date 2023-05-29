from flask import Flask, request, send_file

app = Flask(__name__)

# 请求例如http://127.0.0.1:8081/vision
@app.route('/vision', methods=['POST'])
def process_video():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return 'No file found in the request', 400

    file = request.files['file']

    # 保存文件到服务器
    file.save('/path/to/save/file.mp4')

    # 在这里进行目标检测的处理，使用 yolov5 或其他相关的代码
    # 处理后的结果保存到 /path/to/result/result.mp4
    result_file_path = '/path/to/result/result.mp4'

    # 返回处理后的结果给客户端
    return send_file(result_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
