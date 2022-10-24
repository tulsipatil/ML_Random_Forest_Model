# Import required python packages

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Import dataset

my_df = pd.read_csv("video_game_data.csv")

# Split the data into input and output objects

x = my_df.drop(["completion_time"], axis = 1)
y = my_df["completion_time"]

# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Instantiate the model object

regressor = RandomForestRegressor(random_state = 42)

# Train the model

regressor.fit(x_train, y_train)

# Access model accuracy

y_pred = regressor.predict(x_test)

prediction_comarison = pd.DataFrame({"actual" : y_test,
                                     "prediction" : y_pred})
r2_score(y_test, y_pred)
