import pathlib
from typing import List

import dspy
import pandas as pd


class DataManager:
    @staticmethod
    def get_data(path: pathlib.Path, debug):

        def clean(df_uncleaned):
            df_uncleaned = df_uncleaned[df_uncleaned['source_id'] != '1']
            return df_uncleaned

        df = pd.read_csv(path, index_col=0)
        df = clean(df)
        df = df[df['full_article'] != '<ERROR: ArticleException>']
        df = df.sample(frac=1, random_state=42)
        return df if not debug else df[:25]

    @staticmethod
    def get_examples(path: pathlib.Path, debug) -> List[dspy.Example]:
        df = DataManager.get_data(path, debug)
        examples = [dspy.Example(id=index, **row) for index, row in df.iterrows()]
        examples = DataManager._set_example_inputs(examples) # set inputs
        return examples

    @staticmethod
    def _set_example_inputs(examples):
        """
        Updates each Example in the dataset to specify the input fields.

        Args:
            examples (list of dspy.Example): The dataset containing Example objects.

        Returns:
            list of dspy.Example: The updated dataset with inputs set.
        """
        updated_dataset = []
        for example in examples:
            updated_example = example.with_inputs(
                'title',
                'description',
                'full_article',
            )
            updated_dataset.append(updated_example)
        return updated_dataset


