from pathlib import Path
import os
import asyncio
import re
import pandas as pd
import yaml
from typing import Dict, Tuple
import logging
from custom_logging import get_logger_with_level

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as google_llm

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Constants + setup
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
log = get_logger_with_level( logging.DEBUG ) # TODO - change logging level if the output is too verbose

def get_secret_key(secret_name: str) -> str:
    """
    TODO: Implement this function by either: 
    (1) setting env vars
    (2) using a secret manager of some kind
    (3) inserting your key into the empty string (iff. you are not committing to a repo)
    """
    if secret_name == "OPENAI_API_KEY":
        api_key = "" or os.environ.get(secret_name) 
    elif secret_name == "ANTHROPIC_API_KEY":
        api_key = "" or os.environ.get(secret_name) 
    elif secret_name == "GOOGLE_AI_STUDIO_API_KEY":
        api_key = "" or os.environ.get(secret_name)
    if not api_key:
        raise ValueError(f"API key not found for {secret_name}")
    return api_key

OPEN_AI_API_KEY = get_secret_key("OPENAI_API_KEY")    
ANTHROPIC_API_KEY = get_secret_key("ANTHROPIC_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = "" # get_secret_key("GOOGLE_AI_STUDIO_API_KEY")

openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
google_llm.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)
google_gemini_pro = google_llm.GenerativeModel('gemini-pro')

TEMPERATURE = 0
MAX_TOKENS = 2048
SEMAPHORE_SIZE = 10 # TODO adjust to change the number of concurrent LLM requests

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Helpers
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def read_file_to_text(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")   
    return text

def replace_placeholders(text: str, replacements: Dict[str, str]) -> str:
    """Replace placeholders in instruction and user input templates"""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


async def generate_openai_response(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        completion = await openai_client.chat.completions.create(
            messages=messages,
            # model="gpt-3.5-turbo-0125",
            model="gpt-4-turbo-2024-04-09",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        log.error(f"Error while requesting OpenAI chat completion: {e}")
        raise e
    
    return str(content)

async def generate_anthropic_response(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        completion = await anthropic_client.messages.create(
            messages=messages,
            # model="claude-3-haiku-20240307",
            # model="claude-3-sonnet-20240229",
            model="claude-3-opus-20240229",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.content[0].text
    except Exception as e:
        log.error(f"Error while requesting Anthropic chat completion: {e}")
        raise e
    
    return str(content)

async def generate_google_response(prompt: str) -> str:
    try:
        completion = await google_gemini_pro.generate_content_async(
            prompt,
            generation_config=google_llm.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            )
        )
        content = completion.text
    except Exception as e:
        log.error(f"Error while requesting Google chat completion: {e}")
        raise e
    return str(content)

def parse_reasoning_and_answer(response_text: str) -> Tuple[str, str]:
    # Split the response text based on the delimiter (three or more dashes)
    parts = re.split(r'-{3,}', response_text)
    
    if len(parts) != 2:
        raise ValueError("Invalid response format. Single delimiter not found.")
    
    # Strip leading/trailing whitespace from reasoning and answer
    reasoning = parts[0].strip()
    answer = parts[1].strip()
    
    return (reasoning, answer)

def extract_winner(response: str) -> int:
    if not response:
        raise ValueError("The input response is empty.")
    
    lines = response.strip().split('\n')
    for line in reversed(lines):
        if 'winner' in line.lower():
            numbers = re.findall(r'\d+', line)
            if numbers:
                winner_number = int(numbers[0])
                if winner_number in {1, 2}:
                    return winner_number
                else:
                    raise ValueError("Invalid winner number: must be 1 or 2.")
            else:
                raise ValueError("No number found in the winner declaration line.")
    
    raise ValueError("No winner declaration found in the response.")

# Map provider names to response generating functions
# Note: with this current setup, you should only be using 2 at a time.
provider_functions = {
    "openai": generate_openai_response,
    "anthropic": generate_anthropic_response,
    # "google": generate_google_response
}
assert(len(provider_functions) == 2)

GET_ANSWER__prompt_template = read_file_to_text(BASE_PATH / "inputs/prompt_templates/get_answer.txt")
PICK_FAVORITE__prompt_template = read_file_to_text(BASE_PATH / "inputs/prompt_templates/pick_favorite.txt")

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## 1. Generate answers to prompts for each LLM provider
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
async def process_generate_answer_task(
    semaphore, generate_response, category_name, task_name, provider_name, prompt
):
    # Generate response with semaphore
    async with semaphore:
        try:
            response = await generate_response(prompt)
        except Exception as e:
            log.error(f"Error generating response for task '{task_name}' and provider '{provider_name}': {e}")
            return (category_name, task_name, provider_name, "ERROR", "ERROR")
    # Extract answer from response
    try:
        reasoning, answer = parse_reasoning_and_answer(response)
        return (category_name, task_name, provider_name, reasoning, answer)
    except Exception as e:
        log.error(f"Error parsing answer for task '{task_name}' and provider '{provider_name}': {e}")
        return (category_name, task_name, provider_name, response, "ERROR")


async def generate_answers(task_prompts_by_category, semaphore):
    tasks = []
    output_columns = [
        "task_category", "task_name", "model_provider_name", "response_reasoning", "response_answer"
    ]
    
    output_file = BASE_PATH / "outputs/responses.csv"
    output_file_exists = os.path.isfile(output_file)

    if output_file_exists:
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=output_columns)

    # Iterate over each provider and task
    for provider_name, generate_response in provider_functions.items():
        for category_name, category_tasks in task_prompts_by_category.items():
            for task_name, task_prompt in category_tasks.items():
                # Generate answer if entry doesn't exist (retries errors)
                if not (
                    (df["task_name"] == task_name) & 
                    (df["model_provider_name"] == provider_name) &
                    (df["response_answer"] != "ERROR")
                ).any():
                    replacements = {
                        "{{PROMPT}}": task_prompt
                    }
                    prompt = replace_placeholders(GET_ANSWER__prompt_template, replacements)
                    task = process_generate_answer_task(
                        semaphore, generate_response, category_name, 
                        task_name, provider_name, prompt
                    )
                    tasks.append(task)

    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results and save to CSV
    new_df = pd.DataFrame(results, columns=output_columns)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(output_file, index=False)


##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## 2. Use both LLM providers to pick favorite answer from the previous step
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
async def process_pick_favorite_task(
    semaphore, generate_response, category_name, task_name, provider_name, prompt, shuffled_providers
):
    # Generate response with semaphore
    async with semaphore:
        try:
            response = await generate_response(prompt)
        except Exception as e:
            log.error(f"Error generating response for task '{task_name}' and provider '{provider_name}': {e}")
            return (category_name, task_name, provider_name, "ERROR", "ERROR")
    # Extract favorite/winner from response
    try:
        winner = extract_winner(response)
        if winner == 1:
            preference_ordering = [shuffled_providers[0], shuffled_providers[1]]
        else:
            preference_ordering = [shuffled_providers[1], shuffled_providers[0]]
        return (category_name, task_name, provider_name, response, preference_ordering, shuffled_providers)
    except Exception as e:
        log.error(f"Error parsing answer for task '{task_name}' and provider '{provider_name}': {e}")
        return (category_name, task_name, provider_name, response, "ERROR", shuffled_providers)


async def pick_favorites(task_prompts_by_category, semaphore):
    tasks = []
    output_columns = [
        "task_category",
        "task_name",
        "model_provider_name_judge",
        "response_reasoning",
        "answer_provider_preference_ordering",
        "answer_provider_input_ordering"
    ]

    input_file = BASE_PATH / "outputs/responses.csv"
    output_file = BASE_PATH / "outputs/pick_favorite.csv"
    
    # Read the DataFrames from the input CSV file
    if os.path.isfile(input_file):
        input_df = pd.read_csv(input_file)
    else:
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Check if the output CSV file already exists
    output_file_exists = os.path.isfile(output_file)
    if output_file_exists:
        output_df = pd.read_csv(output_file)
    else:
        output_df = pd.DataFrame(columns=output_columns)

    # Iterate over each task
    for category_name, category_tasks in task_prompts_by_category.items():
        for task_name, task_prompt in category_tasks.items():
            task_rows = input_df[input_df["task_name"] == task_name]
            provider_names_all = list(provider_functions.keys())

            # Shuffle the order of the model provider names to get both permutations
            for shuffled_providers in [provider_names_all, provider_names_all[::-1]]:
                candidate_answers = []
                for provider in shuffled_providers:
                    candidate_answer = task_rows[task_rows["model_provider_name"] == provider]["response_answer"]
                    if not candidate_answer.empty:
                        candidate_answers.append(candidate_answer.iloc[0])
                    else:
                        log.warning(f"No entry for model_provider_name: {provider} for task: {task_name}")
                if not len(candidate_answers) == 2:
                    break
                
                formatted_answers = ""
                answer_tokens = [str(i+1) for i in range(len(candidate_answers))]
                for answer_token, answer in zip(answer_tokens, candidate_answers):
                    formatted_answers += f"#######\n"
                    formatted_answers += f"Candidate Answer {answer_token}: "
                    formatted_answers += f"{answer}\n\n"
                    formatted_answers += f"#######\n"

                replacements = {
                    "{{PROMPT}}": task_prompt,
                    "{{ALL_ANSWERS_FORMATTED}}": formatted_answers,
                }
                prompt = replace_placeholders(PICK_FAVORITE__prompt_template, replacements)
                shuffled_providers_str = str(shuffled_providers)
                
                for provider_name, generate_response in provider_functions.items():
                    # Check if the combination of task and provider already exists in the output DataFrame (retries errors)
                    if not (
                        (output_df["task_name"] == task_name) & 
                        (output_df["model_provider_name_judge"] == provider_name) &
                        (output_df["answer_provider_input_ordering"].apply(lambda x: ','.join(x) if isinstance(x, list) else x) == shuffled_providers_str) &
                        (output_df["answer_provider_preference_ordering"] != "ERROR")
                    ).any():
                        task = process_pick_favorite_task(
                            semaphore, generate_response, category_name, 
                            task_name, provider_name, prompt, shuffled_providers
                        )
                        tasks.append(task)
                        

    # Run tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results and save to CSV
    new_df = pd.DataFrame(results, columns=output_columns)
    updated_df = pd.concat([output_df, new_df], ignore_index=True)
    updated_df.to_csv(output_file, index=False)


##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Run main
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
async def main():
    task_prompts_by_category = load_yaml(BASE_PATH / 'inputs/tasks.yaml')
    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    
    ## Generate answers
    await generate_answers(task_prompts_by_category, semaphore)
    
    ## Judge answers
    await pick_favorites(task_prompts_by_category, semaphore)

asyncio.run(main())