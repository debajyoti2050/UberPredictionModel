from numpy import result_type
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lr_model.pickle', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	Price = int(request.form['Price per week'])
	Population = int(request.form['Population'])
	Monthlyin = int(request.form['Monthly Income'])
	Appm = int(request.form['Average Parking Per Month'])
	
	final_features = pd.DataFrame([[Price, Population, Monthlyin,Appm]])
	
	predict = model.predict(final_features)
	
	
	return render_template('index.html', prediction_text='Number of Weekly Riders are : {}'.format(f"{round(predict[0])}"))
	
if __name__ == "__main__":
	app.run(debug=True)
