import re
from utils.llm import LLM
from utils.logger import Logger


ANSWER_REGEX = r'\\boxed{((?:\\[a-z]+|{[^{}]*}|[^{}])+)}'


def ask_llm_aime(
    llm: LLM,
    problem: str,
    prompt: str,
    max_tokens: int,
    verbose: bool = False,
    timeout: float = 60*10
) -> tuple[int | None, str | None, int | None]:
    prompt = f'{problem}\n\n{prompt}'

    response_text, response_tokens = llm.get_answer(prompt, max_tokens, timeout)

    if not response_text:
        return None, None, None
    
    try:
        match = re.search(ANSWER_REGEX, response_text)
        if match:
            answer = int(match.group(1))
            return answer, response_text, response_tokens
        else:
            Logger.warning('ask_llm_aime', f'The LLM response was not found', verbose)
            return None, response_text, response_tokens
    except Exception as e:
        Logger.warning('ask_llm_aime', f'The LLM response was not an integer: {e}', verbose)
        return None, response_text, response_tokens
