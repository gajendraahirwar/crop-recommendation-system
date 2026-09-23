import pandas as pd
import pickle

with open("crop_model.pkl", "rb") as file:
    model = pickle.load(file)

    sample = pd.DataFrame([{'N': 90, 'P': 42, 'K': 43, 'temperature': 20.879743, 'humidity': 82.002744, 'ph': 6.502985, 'rainfall': 202.935536}])
    prediction = model.predict(sample)

    print("recommended crop for the given soil and weather conditions is:", prediction[0])