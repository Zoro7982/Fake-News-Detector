import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df_fake=pd.read_csv("fake.csv")
df_true=pd.read_csv("true.csv")
df_fake['label'] = 0 
df_true['label'] = 1
df=pd.concat([df_fake, df_true], ignore_index=True)
