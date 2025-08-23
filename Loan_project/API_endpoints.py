### SOURCE CODE
import modelbit, sys
import random

# main function
def predict_weather(days_from_now: int):
    prediction = random.choice(["sunny", "cloudy", "just right"])
    return {
        "weather": prediction,
        "message": f"In {days_from_now} days it will be {prediction}!"
    }

# to run locally via git & terminal, uncomment the following lines
if __name__ == "__main__":
result = predict_weather(...)
print(result)


### CURL SINGLE 
### Option 1
curl -s -XPOST "https://junielgavilansanchez.us-east-1.aws.modelbit.com/v1/predict_weather/latest" -d \
'{"data": "predict_weather(days_from_now=2)"}' | json_pp

### Option 2
curl -s -XPOST "https://junielgavilansanchez.us-east-1.aws.modelbit.com/v1/predict_weather/latest" -d \
'{"data": "2"}' | json_pp

### CURL BATCH 
curl -s -XPOST "https://junielgavilansanchez.us-east-1.aws.modelbit.com/v1/predict_weather/latest" -d '{"data": [[1, days_from_now], [2, days_from_now]]}' | json_pp

### PYTHON SINGLE
modelbit.get_inference(
  region="us-east-1.aws",
  workspace="junielgavilansanchez",
  deployment="predict_weather",
  data=days_from_now
)

### PYTHON BATCH 
modelbit.get_inference(
  region="us-east-1.aws",
  workspace="junielgavilansanchez",
  deployment="predict_weather",
  data=[[1, days_from_now], [2, days_from_now]]
)
