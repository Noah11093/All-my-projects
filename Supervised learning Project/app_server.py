from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np
from AIProject4Api import SupervisedLearning
# from AI_Project_Class import SupervisedLearning


app = Flask(__name__)

# @app.before_first_request
# def before_first_request():
#     load_model()

# def load_model():
#     global model
#     model = SupervisedLearning()
#     print('model loaded!')


@app.route('/lr')
def hello():
    model.setLogisticRegression()
    return jsonify({'score' : model.getscore() })

@app.route('/svc')
def testme():
    model.setSVC()
    return jsonify({'score': model.getscore() })

@app.route('/add/<int:a>/<int:b>')
def add(a, b):
    result = a + b
    return jsonify({'result': result})

@app.route('/predict/<int:a>', methods=['GET'])
@cross_origin()
def predict(a):
    result = model.get_result(a)
    #result = {}
    return jsonify({'result': result})

@app.route('/concat', methods=['GET'])
def concat():
    str = request.query_string
    print(str)
    return jsonify({'result': 'request.query_string'})

@app.route('/print-plot')
def plot_png():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   xs = np.random.rand(100)
   ys = np.random.rand(100)
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

if __name__ == '__main__':
    global model
    model = SupervisedLearning()
    model.evaluation()
    print('model loaded!')
    app.run(host='0.0.0.0',port=8080,debug=True)
    # app.run(debug=True)
    