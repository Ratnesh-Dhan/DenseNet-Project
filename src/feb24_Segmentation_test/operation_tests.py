import json

with open("../../Datasets/testDataset/meta.json", 'r') as file:
    data = json.load(file)

print(len(data['classes']))
titles = [id['title'] for id in data['classes']]
print(titles)