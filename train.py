import numpy as np
import pandas as pd

"""
Data:
The text has been broken down into small groups of words called "nGrams" (like "the cat" or "running fast").
These nGrams are lined up in a file called "Train.csv", with each row representing one nGram.

Features:

For each nGram, they've collected 145 pieces of information called "features".
These features are like details that describe the nGram, such as:
Content: A secret code representing the actual text.
Parsing: What kind of nGram it is (numbers, letters, both).
Spatial: Where it's located in the text and how long it is.
Relational: What the text around it looks like.
These features are saved in two files: "train.csv" and "test.csv".
Labels:

The "labels" are like tags or categories for each nGram.
They tell you which groups the nGram might belong to.
An nGram can belong to multiple groups, so it's a "multilabel" problem.
"""

