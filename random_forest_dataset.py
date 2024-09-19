import pandas as pd

# Load data set
weather_data = pd.read_csv('C:\Users\Joshua\Desktop\python_ai\sklearn_models\archive\IDCJAC0010_066062_1800_Data.csv')

# Display first row
print(weather_data.head())

# Access features
X = weather_data.drop('target_column', axis=1)
y = weather_data['target_column']

