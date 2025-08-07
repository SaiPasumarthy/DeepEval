from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric
from custom_llm import CustomModel
from dotenv import load_dotenv
import os

def test_case():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('OPEN_API_KEY')

    if not api_key:
        print("OPEN AI Error")
        raise ValueError("Please set OPEN_API_KEY environment variable.")
    
    print(f"API KEY : {api_key}")
    obj = CustomModel(api_key=api_key)
    custom_model = obj.invoke()

    # correctness_metric = GEval(
    #     name="Correctness",
    #     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    #     threshold=0.5
    # )
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        # actual_output="You have 30 days to get a full refund at no extra cost.",
        actual_output="We offer a 30-day full refund at no extra costs.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    metric = AnswerRelevancyMetric(model=custom_model)
    assert_test(test_case, [metric])