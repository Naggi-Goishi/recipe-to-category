import csv
import numpy as np
import MeCab
import random
import copy
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

CATEGORY = {
  'japanese': 1,
  'western':  2,
  'chinese':  3,
  'french':   4,
  'italian':  5,
  'spanish':  6,
  'asian':    7,
  'ethnic':   8,
  'dessert':  9
}

def to_dense(data):
  tmp = dictionary.doc2bow(data)
  return list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])

def read_from_csv():
  with open('recipes_view.csv') as csvfile:
    recipes = csv.DictReader(csvfile)
    data = {
      CATEGORY['japanese']: [],
      CATEGORY['western']:  [],
      CATEGORY['chinese']:  [],
      CATEGORY['french']:   [],
      CATEGORY['italian']:  [],
      CATEGORY['spanish']:  [],
      CATEGORY['asian']:    [],
      CATEGORY['ethnic']:   [],
      CATEGORY['dessert']: []
    };
    for row in recipes:
      data[int(row['recipe_kind_id'])].append(row['name'])

  return data


dictionary = corpora.Dictionary.load_from_text('recipe_dictionary.txt')
data       = read_from_csv();

estimator = RandomForestClassifier()

train_dataset = []
train_labels  = []

for label, dataset in data.items():
  for _ in range(0, 10):
    train_dataset.append(to_dense(random.sample(dataset, 20)))
    train_labels.append(label)

file = open('test_italian.txt', 'r')
tmp_dataset = file.readlines()
all_test_dataset = []
test_dataset     = []
test_labels      = []

for name in tmp_dataset:
  all_test_dataset.append(name.rstrip())

for label, dataset in data.items():
  for _ in range(0, 10):
    test_dataset.append(to_dense(random.sample(dataset, 20)))
    test_labels.append(label)

for _ in range(0, 2):
  test_dataset.append(to_dense(random.sample(all_test_dataset, 20)))
  test_labels.append(CATEGORY['italian'])

estimator.fit(train_dataset, train_labels)

print(estimator.score(test_dataset, test_labels))
print(estimator.predict(test_dataset))
