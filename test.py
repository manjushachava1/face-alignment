import base64
import requests

file="Baby_Face.jpg"
with open(file, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
# print(encoded_string)

results = requests.get("http://localhost:8080" ,params={'img':encoded_string})
print(results)
