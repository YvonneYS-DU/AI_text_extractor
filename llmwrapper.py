import json
import re
import asyncio
from datetime import datetime



from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

class Agents:
    def __init__(self):
        self.agents = []
  
    @staticmethod
    def _image_prompts():
        image_prompts = HumanMessagePromptTemplate.from_template(
            [{'image_url': {'path': '{image_path}', 'detail': '{detail_parameter}'}}]
        )
        return image_prompts
    
    @staticmethod
    def _image_prompts_base64():
        image_prompts = HumanMessagePromptTemplate.from_template(
            [{'image_url': {"url": "data:image/jpeg;base64,{base64_image}", 'detail': '{detail_parameter}'}}]
        )
        return image_prompts
    
    @staticmethod
    def _text_prompts(text_prompt_template = 'text prompt template'):
        text_prompts = HumanMessagePromptTemplate.from_template(
            [{'text': text_prompt_template}]
        )
        return text_prompts
    
    @staticmethod
    def lc_prompt_template(text_prompt_template = 'text prompt template', image_prompt_template = True):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                Agents._text_prompts(text_prompt_template),
                *([Agents._image_prompts_base64()] if image_prompt_template else []),
            ])
        return chat_prompt_template
    
    @staticmethod
    def llm_model(
            model: str = 'openai', 
            model_name: str = 'gpt-4o', 
            temperature: float = 0.7, 
            api_key: str = 'api_key', 
            streaming: bool = True
    ):
        # crate llm model object
        if model == 'openai':
            llm = ChatOpenAI(
                model_name = model_name,
                temperature = temperature,
                api_key = api_key,
                streaming = streaming
            )
            print(model, model_name, temperature, api_key, streaming)
            return llm     # return llm model object
            
        else:
            raise ValueError('Model configuration error, check the lambda env config whether in the langchain model list.')
    
    @staticmethod
    def agent_chain_create(model, text_prompt_template='prompt tamplate string', image_prompt_template = True, parameters=False):  # llm model chreate by llm_model, prompt template string, choose to PRINT the PromptTamplate parameters or not
        LC_prompt_template = Agents.lc_prompt_template(text_prompt_template = text_prompt_template, image_prompt_template = image_prompt_template)
        llm = model
        output_parser = StrOutputParser()
        if not parameters:
            chain = LC_prompt_template | llm | output_parser      #return chain: "prompt template | llm model"
            return chain
        else:
            parameters = Agents._extract_prompts_parameters(text_prompt_template)  # list of parameters
            chain = LC_prompt_template | llm | output_parser
            print("Parameters:", parameters)
            return chain, parameters
        
    @staticmethod
    def chain_stream_generator(chain, dic): # gnerate response in stream, to generate respoonse, CHAIN(template, model) and DIC of parameters are required
        for chunk in chain.stream(dic):
            yield chunk.content
            
    @staticmethod
    async def chain_batch_generator_async(chain, dic):
        """Generate response in batch. To generate response, CHAIN(template, model) and DIC of parameters are required."""
        print("taks start at:", datetime.now())
        response = await chain.ainvoke(dic)
        return response
        
    @staticmethod
    def sub_agent(llm, sub_agent_prompt_dic): # create sub agents
        """
        a dict of sub agents with name and prompt template string
        """
        chains = {}
        for key, value in sub_agent_prompt_dic.items():
            chain_name = key
            prompt_template = Agents.lc_prompt_template(value)
            chains[chain_name] = prompt_template | llm
        return chains


class API_unpack:
    """
    this is the resolver of api passed to call llm.
    """


    @staticmethod
    def _PromptImporter(prompt_template='prompt tamplate string'):
        """
        get the prompt template object.
        """
        warning = "[WARNING]: please use [Agents.agent_chain_create(__, prompt_tamplate = 'prompt tamplate string')], this is a developer feature."
        return warning, Agents.lc_prompt_template(prompt_template=prompt_template)
    
    def model_config(config_dic, required_keys=['model', 'model_name', 'temperature', 'api_key', 'streaming']):
        """
        may replaced by lambda env config.
        """
        default_config = {
            'model': 'openai',
            'model_name': 'gpt-4',
            'temperature': 0.7,
            'api_key': 'api_key',
            'streaming': True
        }
        final_config = {}
        # replace the default config with the preferred config
        for key in required_keys:
            if key in config_dic:
                final_config[key] = config_dic[key]
            else:
                final_config[key] = default_config[key]
        return final_config
    
    def get_prompt_name(json_from_api):
        """
        Get the prompt name from the api response.
        """
        try:
            # Json data from API
            if isinstance(json_from_api, str):
                json_data = json.loads(json_from_api)
            else:
                json_data = json_from_api

            # Get the prompt_name from the API response
            for item in json_data.get('prompt', []):
                if item.get('type') == 'sys':
                    return item.get('prompt_name', None)
        
            # If no prompt_name found
            print("Error: prompt['type']='sys' not found, or prompt_name not found in the API response.")
            return None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}.")
            return None

    def data_parameters(data_dic, required_list='prompt_template'):
        required_list = Agents._extract_prompts_parameters(required_list)
        for key in required_list:
            if key not in data_dic:
                raise ValueError(f"Missing key: {key}.")
        return {key: data_dic[key] for key in required_list}

'''class API_unpack:

    @staticmethod'''