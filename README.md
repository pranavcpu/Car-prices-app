#Used Car Price Estimator (India)

Predicts a fair selling price for a second-hand car in India from its age, kilometres driven, fuel type and other details.

Live demo: https://car-prices-app-bkwdnx4sdqbsezwfgdotth.streamlit.app/ (Free apps on Streamlit Community Cloud go to sleep after a few days. If you see a "Wake up" button, click it and wait about a minute.)

Notebook: https://colab.research.google.com/drive/1Esbztw_sBQbgwpnd6gQzZEYDBxnbnuov?usp=sharing

The problem:

Buyers and sellers of used cars often don't know what a fair price is. This project predicts a selling price from the car's details, so both sides have a starting point.

Data:

CarDekho-style used car dataset for India: 15,411 listings, 14 columns.
Target: selling_price in rupees.

Features: brand, vehicle age, km driven, seller type, fuel type, transmission, mileage, engine size, max power, seats.

Cleaning

Step	Rows removed	Rows left
Start		15,411
Dropped columns that add nothing (Unnamed: 0 row number, car_name, model), then removed duplicate rows	168	15,243
Removed impossible records: over 100,000 km per year, over 500,000 km in total, or km_driven exactly equal to selling_price (a likely data entry error)	32	15,211

Other cleaning: fixed the brand spelling ISUZU to Isuzu. There were no missing values.

Method
Log-transformed the target (log1p of price), because prices are right-skewed (a few very expensive cars). Predictions are converted back to rupees with expm1.
One-hot encoded the text columns (43 feature columns in total).
Split 80/20 into 12,168 training rows and 3,043 test rows (random_state=42).
Trained a Linear Regression baseline and a Random Forest (100 trees).
Scored both on the test set, in rupees.
Results (test set, in rupees)
Model	MAE	R²	Typical (median) error
Linear Regression (baseline)	₹147,570	0.738	13.0%
Random Forest	₹108,486	0.730	10.0%

The Random Forest cut the average error by about 26% and its typical prediction is within about 10% of the real price. Its R² is slightly lower than the baseline, because R² is very sensitive to a few large misses on expensive cars (see below).

Most important features (Random Forest): max_power (about 0.64), vehicle_age (about 0.22), engine (about 0.04), km_driven (about 0.035).

Where the model is accurate and where it isn't
Actual price band	Test cars	Average error (MAE)
under ₹3 lakh	427	₹39,693
₹3-6 lakh	1,210	₹55,579
₹6-10 lakh	885	₹83,233
₹10-20 lakh	360	₹178,744
above ₹20 lakh	161	₹670,268

Limitations:

Luxury cars are poorly predicted. Error grows sharply with price. The 161 test cars above ₹20 lakh (about 5% of the test set) produce about a third of the total error.
Rare brands have very few examples. For instance, the dataset has only 3 Bentleys, all 9-10 years old. A Random Forest cannot predict far outside the prices it has seen, so a 5-year-old Bentley gets a price of about ₹10 lakh, far below the real value.
Treat the estimate as a starting point. The model is most reliable for everyday cars under about ₹10 lakh.
The data has no city, car condition, accident history or number of owners, all of which affect real prices.
One dataset, one train/test split, and no hyperparameter tuning.
How the app works

app.py is a Streamlit app. It loads the saved Random Forest, collects the car details in a form, turns them into the same 43 columns the model was trained on, predicts the log price, and converts it back to rupees.

Run it locally
pip install -r requirements.txt
streamlit run app.py

Use Python 3.13 and the same scikit-learn version the model was saved with (see requirements.txt).

Files:

app.py: the Streamlit app
car_price_model.pkl: the trained Random Forest
options.json: dropdown options for the form
requirements.txt: packages the app needs
Ideas for next steps
Group rare brands into "Other" and compare the scores.
Tune the Random Forest.
Add the car model name as a feature.
Show a warning in the app for very high-power cars, where the model is less reliable.
