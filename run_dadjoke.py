import os
import json
import logging
import datetime
import time
import yaml


from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT
from langchain_community.utilities import Requests
from langchain_openai import OpenAI

logger = logging.getLogger()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ['gpt_model'] = config['gpt_model']


    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    scenario = 'dadjoke'

    with open(f"specs/{scenario}_oas.json") as f:
        raw_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_api_spec, only_required=False)

    requests_wrapper = Requests(headers={'Accept': 'application/json'})

    llm = OpenAI(model_name=os.environ['gpt_model'], temperature=0.0, max_tokens=700)

    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)


    query_example = "Get me a dad joke"
    # print(f"Example instruction: {query_example}")
    # query = input("Please input an instruction (Press ENTER to use the example instruction): ")
    query = ''
    if query == '':
        query = query_example

    logger.info(f"Query: {query}")

    start_time = time.time()
    rest_gpt.run(query)
    logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
