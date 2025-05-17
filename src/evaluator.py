import re
from utils.llm import LLM
from utils.logger import Logger


ANSWER_REGEX = r'\\boxed{((?:\\[a-z]+|{[^{}]*}|[^{}])+)}'


def ask_llm_aime(llm: LLM, problem: str, prompt: str, max_tokens: int) -> int | None:
    prompt = f'{problem}\n\n{prompt}\n\n/no_think'

    response = llm.get_answer(prompt, max_tokens)

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