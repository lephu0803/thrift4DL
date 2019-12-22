import json

x = {
    "hello": "world",
    "name": [1, 2, 3, 4]
}

y = json.dumps(x)
print(y)
print(type(y))