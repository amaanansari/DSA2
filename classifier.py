
import pathlib
import pickle
import time
from typing import Literal

import dspy
import pandas as pd
from tqdm import tqdm

from src.dataloader import DataManager
from src.detector import BaseDetectorSignature, Detector
from src.util import LanguageModel

# CONSTANTS
DEBUG: bool = False
SEED: int = 42
API: Literal['lambda', 'openai'] = 'lambda'
MAX_TOKEN: int = 300
PATH = pathlib.Path("data") / "combined_articles_with_sentiment.csv"

lm_wrapper = LanguageModel(max_tokens=MAX_TOKEN, service=API)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm_wrapper.lm, rm=colbertv2_wiki17_abstracts)

def single_example(example: dspy.Example, detector: dspy.Module) -> dspy.Example:
    result = detector(example.title, example.description, example.full_article)
    if result is not None:
        example.label = result[1]
        example.explanation = result[0]
        example.retrieved_context = result[2]
    else:
        example.label = None
        example.explanation = None
        example.retrieved_context = None
    return example


if __name__ == "__main__":
    # Load training and test sets
    exit(-1) # safeguard
    start = time.time()
    examples = DataManager.get_examples(PATH, debug=DEBUG)
    examples = examples[:2]

    model = Detector()
    for example in tqdm(examples, desc="Processing Examples"):
        single_example(example, model)
    df = pd.DataFrame([x.toDict() for x in examples])
    df.to_csv(pathlib.Path("data") / pathlib.Path("subset3_articles_with_sentiment_and_classification.csv"))
    with open(pathlib.Path("data") / pathlib.Path("subset3_articles_with_sentiment_and_classification.pkl"), "wb") as file:
        pickle.dump(examples, file)

    end = time.time()
    usage = lm_wrapper.get_usage()
    print(f"Usage cost (in cents) about {usage[2]}, Input Tokens: {usage[0]}, Output Tokens {usage[1]}" )
    print("Time taken (in seconds)", end - start)
    print("Examples processed: ", len(examples))