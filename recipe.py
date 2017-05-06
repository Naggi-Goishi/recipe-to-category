# coding=UTF-8
import csv
import numpy as np
import MeCab
import random
import copy
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

def to_dense(data):
  tmp = dictionary.doc2bow(data)
  return list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])

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


dictionary = corpora.Dictionary.load_from_text('recipe_dictionary.txt')
data       = read_from_csv();

for recipe_kind in data:
  data[recipe_kind] = to_dense(data[recipe_kind])

estimator = RandomForestClassifier()

train_dataset = []
train_labels  = []

for labels, dataset in data.items():
  train_dataset.append(dataset)
  train_labels.append(labels)


estimator.fit(train_dataset, train_labels)
print(estimator.predict(to_dense(['親子丼', '卵焼き', '天ぷら', '生姜焼き'])))
