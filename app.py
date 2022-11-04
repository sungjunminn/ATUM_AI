import json
from Module.main import test1
from Module.main2 import test2
from Module.ATUM_NLP import toNLP
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/recommendMenu', methods=['GET'])
def recommendMenu():
    # data = request.data.decode("utf-8").replace("'", '"')
    # jsonData = json.loads(data)
    # print(jsonData)
    # print(jsonData['projectId'], jsonData['ruleId'])
    pID = request.args.get('projectId')
    rID = request.args.get('ruleId')
    a = test1(pID, rID)
    print(a)

    return a

@app.route('/test', methods=['GET'])
def test():
    return "abcdefg"


@app.route('/chatAI', methods=['POST'])
def chatAI():
    if request.method == 'POST':
        data = request.data.decode("utf-8").replace("'", '"')
        jsonData = json.loads(data)
        print(jsonData)
        pID = jsonData['projectId']
        rID = jsonData['ruleId']
        question = jsonData['input']
        print("question : ", question)
        test_a = toNLP(question, pID, rID)
        return test_a
    c = '11'
    return c


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
