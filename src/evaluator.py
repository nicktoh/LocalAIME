import re
from utils.llm import LLM
from utils.logger import Logger


ANSWER_REGEX = r'\\boxed{((?:\\[a-z]+|{[^{}]*}|[^{}])+)}'


def ask_llm_aime(
    llm: LLM,
    problem: str,
    prompt: str,
    max_tokens: int,
    qwen3_nothink: bool = False,
    verbose: bool = False,
    timeout: float = 60*10
) -> tuple[int | None, str | None]:
    prompt = f'{problem}\n\n{prompt}'

    if qwen3_nothink and 'qwen3' in llm.model.lower():
        prompt += '\n\n/no_think'
        Logger.info('ask_llm_aime', 'Appending /no_think to prompt for Qwen3', verbose)

    response = llm.get_answer(prompt, max_tokens, timeout)

    if not response:
        return None, None
    
    try:
        match = re.search(ANSWER_REGEX, response)
        if match:
            answer = int(match.group(1))
            return answer, response
        else:
            Logger.warning('ask_llm_aime', f'The LLM response was not found', verbose)
            return None, response
    except Exception as e:
        Logger.warning('ask_llm_aime', f'The LLM response was not an integer: {e}', verbose)
        return None, response