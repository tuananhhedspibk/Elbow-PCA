# -*- coding: utf-8 -*-

from pyvi.pyvi import ViTokenizer
from settings import *

import re
import os
import math
import gensim
import numpy as np

dir = os.path.dirname(__file__)

def handle_string(input_string):
  input_string = input_string.decode("utf-8").lower().encode("utf-8")

  for pattern in REDUNDANT_STRING_PATTERN:
    input_string = re.sub(pattern, "", input_string)

  for symbol in REDUNDANT_SYMBOL:
    if symbol == "\\n":
      input_string = input_string.replace(symbol, ". ")
    else:
      input_string = input_string.replace(symbol, "")

  return input_string

def is_forbidden_string(input_string):
  pattern_under_score = re.compile("_{2,}")
  pattern_minus = re.compile("-{2,}")
  pattern_multiple = re.compile("\*\s\*\s\\\s(\-|\s)+\*\s\*")

  if len(input_string) == 0:
    return True
  elif pattern_under_score.match(input_string):
    return True
  elif pattern_minus.match(input_string):
    return True
  elif pattern_multiple.match(input_string):
    return True
  return False

def pyviConvert(input_str):
  input_str = input_str.decode("utf-8", errors="replace")
  input_str = ViTokenizer.tokenize(input_str)
  input_str = input_str.encode("utf-8")
  return input_str

def write_file(output_file, data, delimiter):
  with open(output_file, "a") as output_file_pt:
    output_file_pt.write(data + delimiter)
  output_file_pt.close()

def load_stopwords():
  stopwords = []
  with open(INPUT_STOPWORDS_FILE_NAME) as fp:
    for line in fp:
      stopwords.append(line.strip("\n"))
  return stopwords

def remove_stopwords(data, stopwords):
  data = data.split()

  filtered_word = []
  for word in data:
    if word.strip() not in stopwords:
      filtered_word.append(word)
  
  data_wtout_sw = " ".join(filtered_word)
  return data_wtout_sw

def handle_file(input_file_path, output_file_path, delimiter, need_to_checked):
  stopwords = load_stopwords()
  training_corpus = []

  ct = 0
  with open(input_file_path) as fp:
    for line in fp:
      if need_to_checked:
        cut_data = line.split("\t")[1]
        cut_id = line.split("\t")[0]
        if len(cut_data.strip()) <= 2 or "file:///" in cut_data:
          continue
        checked_line = handle_string(cut_data)
        if is_forbidden_string(checked_line):
          continue
        outputPyVi = pyviConvert(checked_line)
        if len(outputPyVi) <= 2:
          continue
      else:
        outputPyVi = pyviConvert(line)
      outputPyVi = remove_stopwords(outputPyVi, stopwords)
      write_file(output_file_path, outputPyVi, delimiter)
      outputPyVi = outputPyVi.split()
      training_corpus.append(gensim.models.doc2vec.TaggedDocument(outputPyVi, [ct]))
      ct += 1

  fp.close()
  return training_corpus

def encode_data(training_corpus):
  if not os.path.isfile(MODEL_FILE_NAME):
    model = gensim.models.doc2vec.Doc2Vec(training_corpus, size=320, window=8, min_count=5, workers=5)
    model.save(MODEL_FILE_NAME)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
  else:
    model = gensim.models.doc2vec.Doc2Vec.load(MODEL_FILE_NAME)

  vectorized_data = []
  for doc in training_corpus:
    vectorized_data.append(model.infer_vector(doc.words))

  np.savetxt(ENCODED_DATA_FILE_NAME, vectorized_data, "%.6f")

if __name__ == "__main__":
  training_corpus = handle_file(os.path.join(dir, INPUT_DATA_FILE_NAME),
    os.path.join(dir, PROCESSED_INPUT_DATA_FILE_NAME), "\n", True)
  encode_data(training_corpus)
