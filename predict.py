import json
import os


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


if not os.path.exists("model.json"):
    print("Error: model.json not found")
    exit()


with open("model.json", "r") as file:
    model_data = json.load(file)


theta0 = model_data["theta0"]
theta1 = model_data["theta1"]

km_mean = model_data["mean"]
km_std = model_data["std"]


try:
    mileage = float(input("Enter mileage: "))

    if mileage < 0:
        print("Mileage cannot be negative")
        exit()

except ValueError:
    print("Invalid input")
    exit()


mileage_normalized = (
    mileage - km_mean
) / km_std


prediction = estimate_price(
    mileage_normalized,
    theta0,
    theta1
)


print(f"Estimated price: {prediction:.2f}")