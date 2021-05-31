import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from pathlib import Path
import pyinputplus as pyip
from datetime import datetime
import pickle
import coremltools

columns = ["Date", "Open", "High","Volume"]

df = pd.read_csv(Path("./data/MSFT.csv"))
# shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train and test
ratio = 0.9
split_row = round(ratio * df.shape[0])
train_df = df.iloc[:split_row]
test_df = df.iloc[split_row:]

print("Training model...")
model_input = []
regression_model = linear_model.LinearRegression()
for d, o, h, v in zip(train_df.Date, train_df.Open, train_df.High, train_df.Volume):
    time = pd.to_datetime(d, infer_datetime_format=True, utc=True)  
        ##time = datetime.strptime(d, '%Y-%m-%d %H:%M:%S-%f')
    epoch_time = pd.to_datetime(datetime(1970, 1, 1), infer_datetime_format=True, utc=True)
    epoch_time2 = (time - epoch_time).total_seconds()
    print(epoch_time2)
    model_input.append([epoch_time2, o, h, v])

X = model_input

Y = train_df["Low"]
regression_model.fit(X, Y)
print("Model trained.")

##std_devs = {a: np.std(train_df[a]) for a in ("Low", "double")}
##print(f"Standard deviations: {std_devs}")

# see how well the model did
pred_y = regression_model.predict(test_df[columns])
real_y = test_df["Low"]
assert len(pred_y) == len(real_y), "Must have same number of predictions"
print(f"performance (r2) score: {r2_score(real_y, pred_y)}")

##input_list = [pyip.inputNum(f"Input {prompt}: ") for prompt in columns]
##pred_price = regression_model.predict([input_list])

def save_models(reg_model, feature_list, target_column):
    def save_apple(model, filename):
        # NOTE: coremltools requires sklearn 0.19.2 or below
        output = coremltools.converters.sklearn.convert(
            model, feature_list, target_column
        )
        output.save(filename)
        return output

    if input("save? y/n ").lower().strip() == "y":
        pickle.dump(reg_model, open(Path("models/reg_model.sav"), "wb"))
        save_apple(reg_model, Path("models/reg_model.mlmodel"))
        print("saved 2 models")
    else:
        print("not saved")
feature_list = [i for i in list(df.columns) if i != "Low"]
input_list = [i for i in list(df.columns) if i == "Low"]
save_models(regression_model, feature_list, input_list)