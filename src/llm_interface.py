"""LLM interface for Ollama."""

from typing import TypeAlias, Iterator
from dataclasses import dataclass, field
import ollama
from ollama import Client

from .config import (
    OLLAMA_BASE_URL, 
    LLM_MODEL, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    QUESTION_PROMPT_TEMPLATE
)
from .vector_store import SearchResult

# Type aliases
Message: TypeAlias = dict[str, str]
Messages: TypeAlias = list[Message]


@dataclass
class LLMInterface:
    """Interface for interacting with Ollama LLM."""
    
    model: str = LLM_MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    system_prompt: str = SYSTEM_PROMPT
    _client: Client | None = None
    
    @property
    def client(self) -> Client:
        """Lazy load Ollama client."""
        if self._client is None:
            self._client = Client(host=self.base_url)
        return self._client
    
    def check_model_available(self) -> bool:
        """Check if the model is available locally."""
        try:
            models_response = self.client.list()
            
            # Handle the new ListResponse type
            if hasattr(models_response, 'models'):
                # New API: models_response.models is a list of model objects
                available_models = [m.name for m in models_response.models if hasattr(m, 'name')]
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Old API: models_response is a dict
                available_models = [m['name'] for m in models_response['models']]
            else:
                print(f"Unknown response type: {type(models_response)}")
                return False
            
            return any(self.model in model for model in available_models)
        except Exception as e:
            print(f"Error checking models: {e}")
            return False
    
    def pull_model(self) -> None:
        """Pull the model if not available."""
        if not self.check_model_available():
            print(f"Pulling model: {self.model}")
            self.client.pull(self.model)
            print(f"Model {self.model} pulled successfully")
    
    def format_context(self, search_results: list[SearchResult]) -> str:
        """Format search results into context string."""
        context_parts: list[str] = []
        
        for i, result in enumerate(search_results, 1):
            pages = ", ".join(map(str, result.page_numbers))
            context_parts.append(
                f"[Source {i}: {result.source_file}, Pages: {pages}]\n"
                f"{result.content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def format_prompt(self, question: str, search_results: list[SearchResult]) -> str:
        """Format the complete prompt with context."""
        context = self.format_context(search_results)
        return QUESTION_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
    
    def generate_response(
        self, 
        question: str, 
        search_results: list[SearchResult],
        stream: bool = False
    ) -> str | Iterator[str]:
        """Generate a response using the LLM."""
        # Ensure model is available
        self.pull_model()
        
        # Format prompt
        prompt = self.format_prompt(question, search_results)
        
        # Create messages
        messages: Messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response
        if stream:
            return self._stream_response(messages)
        else:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            
            # Handle different response types
            if hasattr(response, 'message'):
                # New API: response is a ChatResponse object
                return response.message.content
            elif isinstance(response, dict) and 'message' in response:
                # Old API: response is a dict
                return response['message']['content']
            else:
                raise ValueError(f"Unknown response type: {type(response)}")
    
    def _stream_response(self, messages: Messages) -> Iterator[str]:
        """Stream response tokens."""
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        )
        
        for chunk in stream:
            # Handle different chunk types
            if hasattr(chunk, 'message'):
                # New API: chunk is an object with message attribute
                if hasattr(chunk.message, 'content'):
                    yield chunk.message.content
            elif isinstance(chunk, dict):
                # Old API: chunk is a dict
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']


@dataclass
class ConversationManager:
    """Manages conversation history."""
    
    history: Messages = field(default_factory=list)
    max_history: int = 10
    
    def add_exchange(self, question: str, answer: str) -> None:
        """Add a Q&A exchange to history."""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        
        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()
    
    def get_formatted_history(self) -> str:
        """Get formatted conversation history."""
        formatted: list[str] = []
        
        for i in range(0, len(self.history), 2):
            if i + 1 < len(self.history):
                q = self.history[i]["content"]
                a = self.history[i + 1]["content"]
                formatted.append(f"Q: {q}\nA: {a}")
        
        return "\n\n".join(formatted)
