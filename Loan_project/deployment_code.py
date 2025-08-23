import random

def predict_weather(days_from_now: int):
    prediction = random.choice(["sunny", "cloudy", "just right"])
    return {
        "weather": prediction,
        "message": f"In {days_from_now} days it will be {prediction}!"
    }

# Add the python_version parameter to your deploy call
mb.deploy(predict_weather, python_version="3.9")  # Choose any version from 3.6-3.12


mb.deploy(
    predict_weather, 
    config={"pythonVersion": "3.9"}  # Or possibly "python_version": "3.11"
)