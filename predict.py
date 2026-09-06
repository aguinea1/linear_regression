import json
import os


def estimate_price(mileage, theta0, theta1):  # hipotesis exacta del subject
    return theta0 + (theta1 * mileage)


# Antes de entrenar, theta0 y theta1 deben valer 0 (asi lo pide el subject),
# asi que el programa no debe petar si aun no existe model.json
theta0 = 0
theta1 = 0

if os.path.exists("model.json"):
    with open("model.json", "r") as file:
        model_data = json.load(file)

    theta0 = model_data["theta0"]
    theta1 = model_data["theta1"]
else:
    print("No model trained yet, using theta0 = 0 and theta1 = 0")


try:
    mileage = float(input("Enter mileage: "))

    if mileage < 0:
        print("Mileage cannot be negative")
        exit()

except ValueError:
    print("Invalid input")
    exit()


# theta0/theta1 ya estan desnormalizados (ver training.py), asi que aqui
# se aplica la formula tal cual sobre el mileage real, sin normalizar nada
prediction = estimate_price(
    mileage,
    theta0,
    theta1
)


print(f"Estimated price: {prediction:.2f}")