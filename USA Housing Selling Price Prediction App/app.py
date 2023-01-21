from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


te = pickle.load(open('Address_te.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
pr=pickle.load(open('UHSPP_pr.pkl','rb'))


@app.route('/')
def hello_world():
	return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():
    address = request.form.get("address")
    address=[address]
    address=pd.DataFrame(address,columns =['Address'])
    address=te.transform(address['Address'])
    address=address.values.tolist()[0][0]

    population = request.form.get("population")
    population = int(population)

    income = request.form.get("income")
    income = int(income)

    age = request.form.get("age")
    age = int(age)

    rooms = request.form.get("rooms")
    rooms = int(rooms)

	

 
    temp_arr=list()
    temp_arr=temp_arr+[income, age, rooms,population,address]

    data=np.array([temp_arr])
    temp_sc=scaler.transform(data)
    pred=pr.predict(temp_sc)[0]
    pred=round(pred, 2)
    print(temp_arr)     
    print(temp_sc)   
    print(pred)



    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
	app.run(debug=True)
