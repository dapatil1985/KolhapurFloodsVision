from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle
import weather
from flask import *
from datetime import datetime, timedelta

app = Flask(__name__, template_folder="template")
app.secret_key = 'dapkey'

model = pickle.load(open("./models/final_model.pkl", "rb"))
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	flag=False
	check=0
	if request.method == "POST":
		if(request.form['r1']=="bydate"):
			# Predict weather for a specific date
			date_input =pd.to_datetime(request.form['choosedate'], format='%Y-%m-%d') 
			
			earliest_date = datetime(2008, 6, 1)
			latest_date = datetime(2024, 10, 30)
			if earliest_date <= date_input <= latest_date:
				if date_input.month < 6 or date_input.month > 10 or (date_input.month == 10 and date_input.day > 31):
					check=1
				else:
					predicted_weather = weather.predict_previous_weather(date_input)
					maxtemp=predicted_weather['tempmax'].values[0]
					humidity=predicted_weather['humidity'].values[0]
					windspeed=predicted_weather['windspeed'].values[0]
					rainfall=predicted_weather['rainfall'].values[0]
					tendaysrainfall=predicted_weather['10_Cumulative Rainfall'].values[0]
					print(predicted_weather)
					flag=True
			else:
				check=1	
			
			totalrain=0
			current_date = datetime.today()
			current_year = datetime.strptime(current_date.strftime('%d-%m-%Y'),'%d-%m-%Y').year
			next_year = current_year + 1
			if (date_input.year == current_year or date_input.year == next_year):
				if date_input.month < 6 or date_input.month > 10 or (date_input.month == 10 and date_input.day > 31):
					check=2
				else:
					check=0
					tendaysrainfall=round(weather.predict_previous_10_days_rainfall(date_input)['cumulative_rainfall'],2)
					weatherdata=weather.predict_weather(date_input)
					maxtemp=round(weatherdata['predictions']['tempmax'],2)
					humidity=round(weatherdata['predictions']['humidity'],2)
					windspeed=round(weatherdata['predictions']['windspeed'],2)
					rainfall=round(weatherdata['predictions']['rainfall'],2)
					flag=True
			else:
				check=2
		else:
			if request.form['maxtemp']!= "" and request.form['humidity']!="" and request.form['windspeed']!= "" and request.form['rainfall'] !="":
				maxtemp = float(request.form['maxtemp'])
				# Rainfall
				humidity = float(request.form['humidity'])
				# Evaporation
				windspeed = float(request.form['windspeed'])
				# Sunshine
				rainfall = float(request.form['rainfall'])
				# Wind Gust Speed
				tendaysrainfall = float(request.form['tendaysrainfall'])

				date_input=""
				flag=True
			else:
				check=3

		if(check==1 and not flag):
			flash("Please enter a date between June 1,2008 and October 31, 2024.")
			flash("OR")
			flash("Please enter a date of current year and next year between 1 June and 31 October")
			return render_template('messages.html')
		elif(check==2 and not flag):
			flash("Please enter a date between June 1,2008 and October 31, 2024.")
			flash("OR")
			flash("Please enter a date of current year and next year between 1 June and 31 October")	
			return render_template('messages.html')
		elif(check==3 and not flag):
			flash("Please provide input to all required fields")	
			return render_template('messages.html')

		
		if flag:
			input_lst = [maxtemp,humidity,windspeed,rainfall,tendaysrainfall]
			if(date_input!=""):
				date_input = pd.to_datetime(date_input, format='%d-%m-%Y').strftime('%d-%m-%Y')
			weatherdetail={'cdate':date_input,'maxtemp':maxtemp,'humidity':humidity,'windspeed':windspeed,'rainfall':rainfall,'tendaysrainfall':tendaysrainfall}
			input_array = np.array(input_lst).reshape(1, -1)
			prediction = model.predict(input_array)
			output = prediction[0]
			if output == 0:
				return render_template("noflood.html", weatherdata=weatherdetail)
			elif output==1:
				return render_template("normalflood.html", weatherdata=weatherdetail)
			elif output==2:
				return render_template("moderateflood.html", weatherdata=weatherdetail)
			elif output==3:
				return render_template("severeflood.html", weatherdata=weatherdetail)
			else:
				return render_template("extremeflood.html", weatherdata=weatherdetail)

	return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)