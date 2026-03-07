import requests

data = {
    "times": [
        22.52,
        21.54,
        21.48,
        22.69,
        21.79,
        22.92,
        22.00,
        23.13,
        22.58,
        22.53,
        22.33,
        22.68,
        22.70,
        22.83
    ],
    "params": {
        "delta": 0.03,
        "champ_p": 2,
        "dual_p": 6
    }
}

r = requests.post('http://localhost:8000/predict', json=data)
j = r.json()

print(j)
print(j['dist'])
print(j['res'])
print(j['trace'])