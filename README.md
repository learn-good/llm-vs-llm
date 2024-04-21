# Purpose
The purpose of this repo is to provide an alternative or supplement to large language model benchmarks. You can control the tasks tested, the prompt structure, and pick the pair of models you would like to test against each other. You can either manually inspect the generated responses and judgements or run summary statistics on the output `pick_favorite.csv` to see which model the judges preferred the most.


# Description
`judge_pair.py` allows you to asynchronously generate responses for designated LLM providers from a set of tasks. After responses are generated, it uses the same two LLM providers to judge the responses that were generated. These are judged in both order permutations. The outputs of both steps are CSVs.

`inputs` contains the tasks (`tasks.yaml`) and prompt templates for the "get answer" and "pick favorite" taks (`prompt_templates/get_answer.txt` and `prompt_templates/pick_favorite.txt`)

`outputs` contains the output CSVs `responses.csv` and `pick_favorite.csv` (omitted due to size). There are also some figures generated from a previous run testing Anthropic's `claude-3-opus-20240229` and OpenAI's `gpt-4-turbo-2024-04-09`

----

# How to ...
### Run
- Use a virtual environment if you wish: `python3 -m venv venv; source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Edit the "TODO"s in `judge_pair.py`
   1. Modify `get_secret_key()` to point to your API keys. You can skip this step if you have your keys loaded as environment variables.
   2. Change the `SEMAPHORE_SIZE` to modify the number of concurrent requests according to your rate limits.
- Run the python script: `python judge_pair.py`

### Modify
- Change the task prompt contents: modify `inputs/tasks.yaml`
- Change the task prompt templates (that have general instructions): modify `inputs/prompt_templates/*.txt`
- Change the API models: modify model name in `generate_openai_response()`, `generate_anthropic_response()`, `generate_google_response()`
- Use only one model for judging (to save on cost): modify the innermost for loop in `pick_favorites()` to use a specific tuple rather than `provider_functions.items()` 
- Test different versions of an API from the same model provider against each other:
  - copy the generating function (e.g. copy `generate_openai_response()` to `generate_openai_response_0125()`)
  - modify the newly copied function to use a different model (e.g. `model = gpt-4-0125-preview`)
  - modify `provider_functions` in `judge_pair.py` so that the name reflects the model (e.g. `"openai-0125": generate_openai_response_0125`).

----

##### Links
Find us here: [learngood.com](https://www.learngood.com).  
YouTube: coming soon