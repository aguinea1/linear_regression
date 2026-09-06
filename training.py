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

# theta0/theta1 se calcularon sobre km normalizado (para que el gradient
# descent converja bien), pero el subject pide la hipotesis
# estimatePrice(mileage) = theta0 + theta1 * mileage sobre el mileage real.
# Por eso los "desnormalizamos" aqui, para no tener que guardar/usar
# mean y std en predict.py y respetar la formula tal cual.
real_theta1 = theta1 / km_std
real_theta0 = theta0 - (theta1 * km_mean / km_std)

print(real_theta0)
print(real_theta1)


model_data = {
    "theta0": real_theta0,
    "theta1": real_theta1
}


with open("model.json", "w") as file:#model.json
    json.dump(model_data, file)


sorted_data = data.sort_values(by="km")#all sorted by km

sorted_km = sorted_data["km"] #only kms sorted

predicted_prices = estimate_price(
    sorted_km,
    real_theta0,
    real_theta1
)


plt.scatter(km, price)

plt.plot(sorted_km, predicted_prices)

plt.xlabel("Kilometers")
plt.ylabel("Price")

plt.title("Linear Regression")

plt.show()