from typing import Tuple, Optional

import dspy

class BaseDetectorSignature(dspy.Signature):
    """Evaluate of the article includes fake news or not. First give a reason for your answer and then provide a final evaluation."""
    title: str = dspy.InputField(desc='Title of the article')
    description: str = dspy.InputField(desc='Short summary of the article')
    full_article: str = dspy.InputField(desc="The full article")
    context = dspy.InputField(desc="Includes pertinent and verified facts to consider.")
    explanation: str = dspy.OutputField(
        desc='Explain your answer briefly')
    label: bool = dspy.OutputField(desc='Final label: False if the article is fake, True otherwise.')

class Detector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.Predict(BaseDetectorSignature)
        self.retrieve = dspy.Retrieve(k=5)

    def forward(self, title, description, full_article) -> Optional[Tuple[str, bool, str]]:
        # Directly pass the inputs to the process method
        # noinspection PyBroadException
        try:
            retrieved_context = self.retrieve(title + ". " + description).passages
            outputs = self.process(
                title=title,
                description=description,
                full_article=full_article,
                context = retrieved_context
            )

            return outputs.completions[0].explanation, outputs.completions[0].label, retrieved_context
        except Exception as e:
            return None


