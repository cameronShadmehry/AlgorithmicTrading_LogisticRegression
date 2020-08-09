import yfinance as yf, pandas as pd, shutil, os, time, glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from get_all_tickers import get_tickers as gt
from ta import add_all_ta_features
from ta.utils import dropna

# List of the stocks we are interested in analyzing. At the time of writing this, it narrows the list of stocks down to 44.
# If you have a list of your own you would like to use just create a new list instead of using this, for example: tickers = ["FB", "AMZN", ...] 
tickers = gt.get_tickers_filtered(mktcap_min=150000, mktcap_max=10000000)

# Check that the amount of tickers isn't more than 2000
print("The amount of stocks chosen to observe: " + str(len(tickers)))

# These two lines remove the Stocks folder and then recreate it in order to remove old stocks. Make sure you have created a Stocks Folder the first time you run this.
shutil.rmtree("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\")
os.mkdir("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\")

# Holds the amount of API calls we executed
Amount_of_API_Calls = 0

# This while loop is reponsible for storing the historical data for each ticker in our list. Note that yahoo finance sometimes incurs json.decode errors and because of this we are sleeping for 2
# seconds after each iteration, also if a call fails we are going to try to execute it again.
# Also, do not make more than 2,000 calls per hour or 48,000 calls per day or Yahoo Finance may block your IP. The clause "(Amount_of_API_Calls < 1800)" below will stop the loop from making
# too many calls to the yfinance API.
# Prepare for this loop to take some time. It is pausing for 2 seconds after importing each stock.

# Used to make sure we don't waste too many API calls on one Stock ticker that could be having issues
Stock_Failure = 0
Stocks_Not_Imported = 0

# Used to iterate through our list of tickers
i=0
while (i < len(tickers)) and (Amount_of_API_Calls < 1800):
    try:
        stock = tickers[i]  # Gets the current stock ticker
        temp = yf.Ticker(str(stock))
        Hist_data = temp.history(period="max")  # Tells yfinance what kind of data we want about this stock (In this example, all of the historical data)
        Hist_data.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\"+stock+".csv")  # Saves the historical data in csv format for further processing later
        time.sleep(2)  # Pauses the loop for two seconds so we don't cause issues with Yahoo Finance's backend operations
        Amount_of_API_Calls += 1 
        Stock_Failure = 0
        i += 1  # Iteration to the next ticker
        print("Importing stock data:" + str(i))
    except ValueError:
        print("Yahoo Finance Backend Error, Attempting to Fix")  # An error occured on Yahoo Finance's backend. We will attempt to retreive the data again
        if Stock_Failure > 5:  # Move on to the next ticker if the current ticker fails more than 5 times
            i+=1
            Stocks_Not_Imported += 1
        Amount_of_API_Calls += 1
        Stock_Failure += 1
print("The amount of stocks we successfully imported: " + str(i - Stocks_Not_Imported))


# These two lines remove the Stocks folder and then recreate it in order to remove old stocks. Make sure you have created a Stocks Folder the first time you run this.
shutil.rmtree("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\")
os.mkdir("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\")

# Get the Y values
list_files = (glob.glob("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\*.csv")) # Creates a list of all csv filenames in the stocks folder
for interval in list_files:
    Stock_Name = ((os.path.basename(interval)).split(".csv")[0])
    data = pd.read_csv(interval)
    dropna(data)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    data = data.iloc[100:]
    close_prices = data['Close'].tolist()
    Five_Day_Obs = []
    thirty_Day_Obs = []
    sixty_Day_Obs = []
    x = 0
    while x < (len(data)):
        if x < (len(data)-5):
            if ((close_prices[x+1] + close_prices[x+2] + close_prices[x+3] + close_prices[x+4] + close_prices[x+5])/5) > close_prices[x]:
                Five_Day_Obs.append(1)
            else:
                Five_Day_Obs.append(0)
        else:
            Five_Day_Obs.append(0)
        x+=1
    y = 0
    while y < (len(data)):
        if y < (len(data)-30):
            ThirtyDayCalc = 0
            y2 = 0
            while y2 < 30:
                ThirtyDayCalc = ThirtyDayCalc + close_prices[y+y2]
                y2 += 1
            if (ThirtyDayCalc/30) > close_prices[y]:
                thirty_Day_Obs.append(1)
            else:
                thirty_Day_Obs.append(0)
        else:
            thirty_Day_Obs.append(0)
        y+=1
    z = 0
    while z < (len(data)):
        if z < (len(data)-60):
            SixtyDayCalc = 0
            z2 = 0
            while z2 < 60:
                SixtyDayCalc = SixtyDayCalc + close_prices[z+z2]
                z2 += 1
            if (SixtyDayCalc/60) > close_prices[z]:
                sixty_Day_Obs.append(1)
            else:
                sixty_Day_Obs.append(0)
        else:
            sixty_Day_Obs.append(0)
        z+=1
    data['Five_Day_Observation_Outcome'] = Five_Day_Obs
    data['Thirty_Day_Observation_Outcome'] = thirty_Day_Obs
    data['Sixty_Day_Observation_Outcome'] = sixty_Day_Obs
    data.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\"+Stock_Name+".csv")
    print("Data for " + Stock_Name + " has been substantiated with technical features.")

Hold_Results = []
list_files2 = (glob.glob("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\*.csv")) # Creates a list of all csv filenames in the stocks folder
for interval2 in list_files2:
    Stock_Name = ((os.path.basename(interval2)).split(".csv")[0])
    data = pd.read_csv(interval2,index_col=0)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    dependents = [data["Five_Day_Observation_Outcome"].to_list(), data["Thirty_Day_Observation_Outcome"].to_list(), data["Sixty_Day_Observation_Outcome"].to_list()]
    data = data.drop(['Five_Day_Observation_Outcome', 'Thirty_Day_Observation_Outcome', 'Sixty_Day_Observation_Outcome', 'Date', 'Open', 'High', 'Low', 'Close'], axis = 1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)  # Standardize our data set
    Hold_Results_Section = []
    p = 0
    for dep in dependents:
        x_train, x_test, y_train, y_test =\
        train_test_split(data, dep, test_size=0.2, random_state=0)
        model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',random_state=0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)  # To get the predicted values
        conf = confusion_matrix(y_test, y_pred)
        if p == 0:
            Hold_Results.append([Stock_Name, "Five_Day_Observation_Outcome", model.score(x_train, y_train),model.score(x_test, y_test),conf[0,0],conf[0,1],conf[1,0],conf[1,1]])
        if p == 1:
            Hold_Results.append([Stock_Name, "Thirty_Day_Observation_Outcome", model.score(x_train, y_train),model.score(x_test, y_test),conf[0,0],conf[0,1],conf[1,0],conf[1,1]])
        if p == 2:
            Hold_Results.append([Stock_Name, "Sixty_Day_Observation_Outcome", model.score(x_train, y_train),model.score(x_test, y_test),conf[0,0],conf[0,1],conf[1,0],conf[1,1]])
        p+=1
    print("Model complete for " + Stock_Name)
df = pd.DataFrame(Hold_Results, columns=['Stock', 'Observation Period', 'Model Accuracy on Training Data', 'Model Accuracy on Test Data', 'True Positives','False Positives',
'False Negative','True Negative'])
df.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Model_Outcome.csv", index = False)