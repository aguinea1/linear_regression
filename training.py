import pandas as pd  # pandas para trabajar con tablas(data.csv)
import matplotlib.pyplot as plt #visualizar datos (grafica)
import json


def estimate_price(mileage, theta0, theta1): #formula de regresion lineal
    return theta0 + (theta1 * mileage)


try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: data.csv not found")
    exit()


km = data["km"]
price = data["price"] #guardar ddatos

km_mean = km.mean() #media de datos
km_std = km.std()

km_normalized = (km - km_mean) / km_std #normalizacion (gradient descent va mal con numeros grandes)

theta0 = 0
theta1 = 0 #el  agoritm empieza desde (0, 0) en el graph

learning_rate = 0.01 #cantidad de mod por iter
iterations = 1000 


for i in range(iterations):

    estimated_price = estimate_price(#predict de price
        km_normalized,
        theta0,
        theta1
    )

    error = estimated_price - price #predit- reality

    tmp_theta0 = learning_rate * error.mean()  #corregir thetas

    tmp_theta1 = learning_rate * (
        error * km_normalized
    ).mean()

    theta0 = theta0 - tmp_theta0
    theta1 = theta1 - tmp_theta1 #acecar mas thetas

    if i % 100 == 0:
        print(i, theta0, theta1)

mse = (error ** 2).mean() #mean squared error(cuanto se aleja)


print(f"MSE: {mse}")
print("\nTraining finished")

print(theta0)
print(theta1)


model_data = { 
    "theta0": theta0,
    "theta1": theta1,
    "mean": km_mean,
    "std": km_std
}


with open("model.json", "w") as file:#model.json
    json.dump(model_data, file)


sorted_data = data.sort_values(by="km")#all sorted by km

sorted_km = sorted_data["km"] #only kms sorted

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