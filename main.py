from dotenv import load_dotenv
import os
from Evaluation.custom_llm import CustomModel

def main():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('OPEN_API_KEY')

    if not api_key:
        raise ValueError("Please set OPEN_API_KEY environment variable.")
    
    obj = CustomModel(api_key=api_key)
    obj.invoke()


if __name__ == "__main__":
    main()