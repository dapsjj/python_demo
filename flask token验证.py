from flask import Flask, g
from flask_httpauth import HTTPTokenAuth

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    "secret-token-1": "John",
    "secret-token-2": "Susan"
}

@auth.verify_token
def verify_token(token):
    g.user = None
    if token in tokens:
        g.user = tokens[token]
        return True
    return False

@app.route('/')
@auth.login_required
def index():
    print(123)
    return "Hello, %s!" % g.user

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)
