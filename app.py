from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
