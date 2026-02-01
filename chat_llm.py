import gc
import logging
from pathlib import Path
from abc import ABC, abstractmethod

import requests
import torch
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from vector_db_query import get_query_db
from utilities import format_citations
from constants import system_message, rag_string
from config import get_config
from cuda_manager import get_cuda_manager

logger = logging.getLogger(__name__)

config = get_config()
contexts_output_file_path = config.root_dir / "contexts.txt"
metadata_output_file_path = config.root_dir / "metadata.txt"

class LLMSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)


def assemble_rag_prompt_and_query(query: str, selected_database: str) -> tuple[str, list]:
    with get_query_db(selected_database) as query_db:
        contexts, metadata_list = query_db.search(query)

    with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
        for metadata in metadata_list:
            output_file.write(f"{metadata}\n")

    with contexts_output_file_path.open('w', encoding='utf-8') as output_file:
        for ctx in contexts:
            output_file.write(f"{ctx}\n\n---\n\n")

    if not contexts:
        raise ValueError("No relevant contexts found.")

    augmented_query = (
        f"{rag_string}\n\n---\n\n"
        + "\n\n---\n\n".join(contexts)
        + f"\n\n-----\n\n{query}"
    )

    return augmented_query, metadata_list


class LLMClient(ABC):

    @abstractmethod
    def create_client(self):
        pass

    @abstractmethod
    def prepare_completion_params(self, messages: list) -> dict:
        pass

    @abstractmethod
    def should_strip_leading_space(self) -> bool:
        pass


class LMStudioClient(LLMClient):

    def __init__(self, config):
        self.config = config

    def create_client(self):
        base_url = self.config.server.connection_str
        return OpenAI(base_url=base_url, api_key='lm-studio')

    def prepare_completion_params(self, messages: list) -> dict:
        return {
            "model": "local-model",
            "messages": messages,
            "stream": True
        }

    def should_strip_leading_space(self) -> bool:
        return True

class OpenAIClient(LLMClient):

    def __init__(self, config, model_override: str = None):
        self.config = config
        self.model_override = model_override

    def create_client(self):
        api_key = self.config.openai.api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key not found in config.yaml.\n\n"
                "Please set it within the 'File' menu."
            )
        return OpenAI(api_key=api_key)

    def prepare_completion_params(self, messages: list) -> dict:
        model = self.model_override or self.config.openai.model
        return {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": True,
        }

    def should_strip_leading_space(self) -> bool:
        return False


class UnifiedLLMChat:
    def __init__(self, client_strategy: LLMClient):
        self.client_strategy = client_strategy
        self.signals = LLMSignals()
        self.config = get_config()

    def ask_llm(self, query: str, selected_database: str):

        try:
            augmented_query, metadata_list = assemble_rag_prompt_and_query(
                query, selected_database
            )

            client = self.client_strategy.create_client()
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": augmented_query}
            ]

            completion_params = self.client_strategy.prepare_completion_params(messages)
            stream = client.chat.completions.create(**completion_params)

            full_response = ""
            first_content = True
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content

                    if first_content and self.client_strategy.should_strip_leading_space():
                        content = content.lstrip()
                        first_content = False

                    self.signals.response_signal.emit(content)
                    full_response += content

            self.signals.response_signal.emit("\n")

            citations = format_citations(metadata_list)
            self.signals.citations_signal.emit(citations)

            get_cuda_manager().safe_empty_cache()
            gc.collect()

            self.signals.finished_signal.emit()

        except Exception as e:
            logger.exception(f"Error in ask_llm: {e}")
            self.signals.error_signal.emit(str(e))
            self.signals.finished_signal.emit()


class LLMChatThread(QThread):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

    def __init__(self, query: str, selected_database: str, client_strategy: LLMClient):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.llm_chat = UnifiedLLMChat(client_strategy)

        self.llm_chat.signals.response_signal.connect(self.response_signal.emit)
        self.llm_chat.signals.error_signal.connect(self.error_signal.emit)
        self.llm_chat.signals.finished_signal.connect(self.finished_signal.emit)
        self.llm_chat.signals.citations_signal.connect(self.citations_signal.emit)

    def run(self):
        try:
            self.llm_chat.ask_llm(self.query, self.selected_database)
        except Exception as e:
            logger.exception(f"Error in LLMChatThread: {e}")
            self.error_signal.emit(str(e))


def create_lm_studio_thread(query: str, selected_database: str) -> LLMChatThread:
    config = get_config()
    client = LMStudioClient(config)
    return LLMChatThread(query, selected_database, client)


def create_openai_thread(query: str, selected_database: str, model_name: str = None) -> LLMChatThread:
    config = get_config()
    client = OpenAIClient(config, model_override=model_name)
    return LLMChatThread(query, selected_database, client)


def is_lm_studio_available():
    try:
        config = get_config()
        base_url = config.server.connection_str
        url = base_url.rstrip('/') + '/models/'
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False