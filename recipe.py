# coding=UTF-8
import csv
import numpy as np
import MeCab
import random
import copy
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

def read_from_csv():
  with open('recipes_view.csv') as csvfile:
    recipes = csv.DictReader(csvfile)
    data = {
      1: [],
      2: [],
      3: [],
      4: [],
      5: [],
      6: [],
      7: [],
      8: [],
      9: []
    };
    for row in recipes:
      data[int(row['recipe_kind_id'])].append(row['name'])

  return data


def random_data(data):
  train_dataset = []
  train_labels = []
  tmpData = copy.deepcopy(data)

  for recipe_kind, dense in tmpData.items():
    train_labels.append(recipe_kind)
    tmp = []
    for _ in range(6):
      tmp.append(dense.pop(random.randint(0, len(dense) - 1 )))
    train_dataset.append(tmp)

  return train_dataset, train_labels

def get_allwords(data):
  all_words = []
  for recipe_kind, names in data.items():
    words = []
    for name in names:
      node = mecab.parseToNode(name)
      while node:
        words.append(node.surface)
        node = node.next

    all_words.append(words)

  return all_words

def to_dense(data):
  tmp = dictionary.doc2bow(data)
  return list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
  


data = read_from_csv()
mecab = MeCab.Tagger("-Ochasen")
all_words = get_allwords(data)

dictionary = corpora.Dictionary(all_words)

for recipe_kind in data:
  data[recipe_kind] = to_dense(all_words[recipe_kind-1])

estimator = RandomForestClassifier()

train_dataset = []
train_labels  = []

for labels, dataset in data.items():
  train_dataset.append(dataset)
  train_labels.append(labels)


estimator.fit(train_dataset, train_labels)
print(estimator.predict(to_dense(['親子丼'])))
