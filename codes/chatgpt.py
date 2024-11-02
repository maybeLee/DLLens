import json
import os
import re
import time
import traceback
import urllib.request
import openai
from utils.utils import count_tokens

class NotEnoughBudgetError(object):
    pass

class ChatGPTCall(object):
    def __init__(self, api_key_file="./data/api.key", model_name="gpt-35-turbo"):
        self.api_key = self.load_api(api_key_file)
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."}]

    @staticmethod
    def load_api(api_key_file):
        if not os.path.exists(api_key_file):
            raise ValueError(
                f"The API Key File Is Not Found: {api_key_file}. Please Create It And Store Your API Key Here.")
        with open(api_key_file, "r") as file:
            api_key = file.read().split("\n")[0]
        return api_key

    @staticmethod
    def calculate_budget(prompt_tokens=0, completion_tokens=0, mode="gpt-35-turbo"):
        if mode in ["gpt-35-turbo", "gpt-3.5-turbo"]:
            # $1.50 / 1M tokens, $2.00 / 1M tokens
            return prompt_tokens / 1000 * 0.0015 + completion_tokens / 1000 * 0.002
        elif mode == "gpt-4o-mini":
            # $0.150 / 1M input tokens, $0.600 / 1M output tokens
            return prompt_tokens / 1000 * 0.00015 + completion_tokens / 1000 * 0.0006
        elif mode == "gpt-4o":
            # $5.00 / 1M input tokens, $15.00 / 1M output tokens
            return prompt_tokens / 1000 * 0.005 + completion_tokens / 1000 * 0.015

    def current_cost(self):
        return self.calculate_budget(self.prompt_tokens, self.completion_tokens, self.model_name)

    def ask_gpt_openai(self, messages, timeout=300, temperature=0.8, num_choices=1, max_token=256) -> [str]:
        sent_token_size = 0
        for conversation in messages:
            for key, value in conversation.items():  # {"role": "user", "content": "xxx"}
                sent_token_size += count_tokens(key) + count_tokens(value)
        if sent_token_size > 2500:
            # to avoid exceed token limit, we SILENTLY reject the large-sized query.
            print(f"Exceed token limit: {sent_token_size}")
            return ["" for i in range(num_choices)]
        if self.model_name == "gpt-35-turbo":
            model_name = "gpt-3.5-turbo"
        else:
            model_name = self.model_name
        openai.api_key = self.api_key
        try:
            res = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_token,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=num_choices,
                timeout=timeout
            )
            self.prompt_tokens += int(res['usage']['prompt_tokens'])
            self.completion_tokens += int(res['usage']['completion_tokens'])
            content_list = []
            for choice in res.choices:
                content_list.append(choice.message.content)
            return content_list
        except openai.error.Timeout as e:
            # Handle timeout error, e.g. retry or log
            print(f"OpenAI API request timed out: {e}")
            time.sleep(60)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except openai.error.APIError as e:
            # Handle API error, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(1)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except openai.error.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            print(f"OpenAI API request failed to connect: {e}")
            time.sleep(1)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            print(f"OpenAI API request was invalid: {e}")
            time.sleep(1)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            print(f"OpenAI API request was not authorized: {e}")
            time.sleep(1)
            return NotEnoughBudgetError
            # return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except openai.error.PermissionError as e:
            # Handle permission error, e.g. check scope or log
            print(f"OpenAI API request was not permitted: {e}")
            time.sleep(1)
            # we terminate the process
            return NotEnoughBudgetError
        except openai.error.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            print("OpenAI API request exceeded rate limit")
            time.sleep(15)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)
        except Exception:
            print(traceback.format_exc())
            print("Error During Querying, Sleep For One Minute")
            time.sleep(5)
            return self.ask_gpt_openai(messages, temperature=temperature, num_choices=num_choices, max_token=max_token)

    def set_context(self, context: list) -> None:
        """
        Set the context for gpt query
        :param context: contexts including previous dialogs, examples.
        :return: None
        """
        if context is None:
            return
        self.messages = context

    def query(self, query, timeout=60 * 5, keep_previous=False):
        """
        The query function for user to send query to the chatGPT
        :param query: the user's prompt
        :param timeout: the timeout for the user's query
        :param keep_previous: The GPT will also consider previous queries to answer the question
        :return: the GPT's response
        """
        if keep_previous is False:
            messages = [
                {"role": "user", "content": f"{query}"},
            ]
            # return self.ask_gpt(messages, timeout)
            return self.ask_gpt_openai(messages, timeout)[0]
        else:
            self.messages.append(
                {"role": "user", "content": f"{query}"}
            )
            # res = self.ask_gpt(self.messages, timeout)
            res = self.ask_gpt_openai(self.messages, timeout)[0]
            self.messages.append(
                {"role": "assistant", "content": f"{res}"}
            )
            return res

    def dialog(self, query, timeout=60 * 5):
        """
        Open a dialog to use ChatGPT, ChatGPT will answer the query by also considering previous content.
        :param query: the user's query in this iteration
        :param timeout: the timeout for ther user's query
        :return: the GPT's response
        """
        return self.query(query, timeout=timeout, keep_previous=True)

    def clear_previous(self):
        self.messages = []

    def interface(self):
        """
        This is the commandline interface to use ChatGPT
        :return: None
        """
        if not os.path.exists("./sessions"):
            os.mkdir("./sessions")
        with open("./sessions/untitled.txt", 'w') as file:
            while True:
                input_query = input('\n\U0001F4AC Your Query: ')
                file.write("User: " + input_query + '\n')
                if input_query.lower() == "exit":
                    break
                res = self.dialog(input_query)
                res = '\n'.join(filter(lambda x: len(x) > 0, res.split('\n')))
                print(f"\n\U0001F4BB ChatGPT: {res}")
                file.write("ChatGPT: " + res + '\n')
        print()
        filename = input("If you want to save this session, please type a filename without extension: ")
        if filename is not None and filename != "":
            os.rename("./sessions/untitled.txt", "./sessions/" + filename + ".txt")

    @staticmethod
    def extract_code(respond):
        if '`' in respond:
            seed = re.search(r"```(.*?)```", respond, re.DOTALL)
            if seed is not None:
                return seed.group(0)[4:-4]
        elif "<code>" in respond:
            seed = re.search(r"<code>(.*?)</code>", respond, re.DOTALL)
            if seed is not None:
                return seed.group(0)[7:-8]
        return None


if __name__ == "__main__":
    chatGPTCall = ChatGPTCall(api_key_file="./data/api_keys/api.key", model_name="gpt-4o-mini")
    chatGPTCall.interface()
