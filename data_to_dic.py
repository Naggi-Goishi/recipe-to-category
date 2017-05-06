import csv
import MeCab
from gensim import corpora

def all_words():
  with open('recipes_view.csv') as csvfile:
    recipes = csv.DictReader(csvfile)
    names   = []
    for row in recipes:
      names.append(row['name'])

  mecab = MeCab.Tagger("-Ochasen")
  words = []

  for name in names:
    node = mecab.parseToNode(name)
    word = []
    while node:
      word.append(node.surface)
      node = node.next

    words.append(word)

  return words

def get_allwords(data):
  mecab = MeCab.Tagger("-Ochasen")
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

dictionary = corpora.Dictionary(all_words())
dictionary.save_as_text('recipe_dictionary.txt')
