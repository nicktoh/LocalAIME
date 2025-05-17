import re
import pandas as pd
import argparse

from tqdm import tqdm

from utils.llm import LLM
from utils.logger import Logger

AIME_DATASET = 'resources/aime2024.parquet'
PROMPT = 'Given the problem above, reply with the number inside \\boxed{} to provide the final answer.'
ANSWER_REGEX = r'\\boxed{((?:\\[a-z]+|{[^{}]*}|[^{}])+)}'
MAX_TOKENS = 8000


def load_aime_dataset() -> list[tuple[int, str, int]]:
    dataset = pd.read_parquet(AIME_DATASET)

    ids = dataset['id'].astype(int).tolist()
    problems = dataset['problem'].tolist()
    answers = dataset['answer'].astype(int).tolist()
    return list(zip(ids, problems, answers))


def ask_llm(llm: LLM, problem: str) -> int | None:
    prompt = f'{problem}\n\n{PROMPT}\n\n/no_think'

    response = llm.get_answer(prompt, MAX_TOKENS)

    if not response:
        return None
    
    try:
        match = re.search(ANSWER_REGEX, response)
        if match:
            answer = int(match.group(1))
            return answer
        else:
            Logger.warning('ask_llm', f'The LLM response was not found')
    except Exception as e:
        Logger.warning('ask_llm', f'The LLM response was not an integer: {e}')
        return None



def main():
    parser = argparse.ArgumentParser(description='Process AIME dataset with specified model.')
    parser.add_argument('--base-url', type=str, required=True, help='Base URL for the OpenAI-compatible API')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to test')
    parser.add_argument('--api-key', type=str, required=False, default='none', help='API key for the OpenAI-compatible API (optional)')
    args = parser.parse_args()

    aime = load_aime_dataset()
    llm = LLM(args.base_url, args.model, args.api_key)

    for id, problem, solution in tqdm(aime, desc='Testing on AIME', ncols=100, unit='problem'):
        llm_solution = ask_llm(llm, problem)

        if not llm_solution:
            tqdm.write(f'{id}: ❕ Missing')
            continue

        if llm_solution == solution:
            tqdm.write(f'{id}: ✅ Correct')
        else:
            tqdm.write(f'{id}: ❌ Wrong')


if __name__ == '__main__':
    main()