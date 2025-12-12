import os
import json

content = ''
with open('error-2023-05-08-0.log') as file_object:
    content = file_object.read()

print(content)

username = 'shen'

with open('username.json', mode='w') as file:
    json.dump(username, file)
