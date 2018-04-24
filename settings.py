# -*- coding: utf-8 -*-

A_IDS_FILE_NAME = "./input/a_ids.txt"
INPUT_DATA_FILE_NAME = "./input/articles.tsv"
PROCESSED_INPUT_DATA_FILE_NAME = "./input/data.txt"
INPUT_STOPWORDS_FILE_NAME = "./input/stopwords.txt"
ENCODED_DATA_FILE_NAME = "./input/data.csv"
VOCAB_FILE_NAME = "./intermediate_output/vocab.txt"
MODEL_FILE_NAME = "./intermediate_output/model.dat"

REDUNDANT_SYMBOL = ["\\t", "\\r", "\\n", "##", "\\", "”",
  "**___________________**", "**", "---", "_", "|", "…",
  "---|---|---|---|---"]

REDUNDANT_PATTERN = ["[a-z]\)", "đ\)", "&amp",
  "(\d+\/)*([A-Z\d\%\-a-z]+)+(\&[a-z]+\;([a-z]+\=[A-Za-z\d]+)+)+\"",
  "([a-z]+\=\"[a-z_]+\"\&[a-z;]+)+",
  "((\)\;)+[a-z\-\:\;\"]+)+", "([a-z]+\=\".*\-)+",
  "(right:[a-z\d]+\;)(color\:[a-z\(\d\,\s]+)"]

REDUNDANT_STRING_PATTERN = ["điều \d{1,2}", "khoản \d{1,2}",
  "chương [ivxlcdm]+", "điểm \d{1,2}",
  "phần thứ [a-zâấưăáảáíờộơ ]+", "mục \d{1,2}",
  "cộng hoà xã hội chủ nghĩa việt nam", "độc lập - tự do - hạnh phúc",
  "\d{1,2}\\\.", "[\w]\)", "đ\)"]

FORBIDEN_SYMBOL_TOPICS = ["_", "%", "|", "\\", ":"]
