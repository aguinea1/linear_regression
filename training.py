import pandas as pd
import matplotlib.pyplot as plt
import json


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: data.csv not found")
    exit()


km = data["km"]
price = data["price"]

km_mean = km.mean()
km_std = km.std()

km_normalized = (km - km_mean) / km_std

theta0 = 0
theta1 = 0

learning_rate = 0.01
iterations = 1000


for i in range(iterations):

    estimated_price = estimate_price(
        km_normalized,
        theta0,
        theta1
    )

    error = estimated_price - price

    tmp_theta0 = learning_rate * error.mean()

    tmp_theta1 = learning_rate * (
        error * km_normalized
    ).mean()

    theta0 = theta0 - tmp_theta0
    theta1 = theta1 - tmp_theta1

    if i % 100 == 0:
        print(i, theta0, theta1)

mse = (error ** 2).mean()

rmse = mse ** 0.5

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print("\nTraining finished")

print(theta0)
print(theta1)


model_data = {
    "theta0": theta0,
    "theta1": theta1,
    "mean": km_mean,
    "std": km_std
}


with open("model.json", "w") as file:
    json.dump(model_data, file)


sorted_data = data.sort_values(by="km")

sorted_km = sorted_data["km"]

sorted_km_normalized = (
    sorted_km - km_mean
) / km_std


predicted_prices = estimate_price(
    sorted_km_normalized,
    theta0,
    theta1
)


plt.scatter(km, price)

plt.plot(sorted_km, predicted_prices)

plt.xlabel("Kilometers")
plt.ylabel("Price")

plt.title("Linear Regression")

plt.show()