from llama_cpp import  Llama
import os
import sys

# Configuration
# Path to the GGUF model we downloaded
from prompt_templates.prompts import PromptManager

# Configuration
# Path to the GGUF model we downloaded
MODEL_PATH = "./models/gemma-2b-it.Q4_K_M.gguf"
CONTEXT_WINDOW = 2048

class PocketLLM:
    def __init__(self):
        print(f"Loading LLM from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file {MODEL_PATH} not found!")
            print("Please run download_models.py first.")
            self.llm = None
            return

        # Initialize Llama model
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=CONTEXT_WINDOW,
            n_threads=4,      # Adjust based on CPU cores
            verbose=False     # Set to True for debug
        )
        print("LLM loaded.")
        
        # Initialize Prompt Manager
        self.prompts = PromptManager()

    def generate_response(self, user_text):
        """
        Generates a response from the LLM.
        """
        if not self.llm:
            return "Error: LLM not loaded."

        # Use PromptManager to build the few-shot context
        messages = self.prompts.construct_messages(user_text)
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,       # Increased for better info
                temperature=0.6,      # Calm/Stable
                stop=["<end_of_turn>", "User:", "\nUser", "<start_of_turn>"] 
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error during inference: {e}")
            return "I'm having trouble thinking."

    # Intent detection is now handled by the LLM's response (Tool Use)
    def check_search_intent(self, user_text):
        """Deprecated: Logic moved to LLM prompt."""
        return False

    def generate_response_with_search(self, user_text, search_context):
        """
        Generates a response using search context.
        """
        if not self.llm:
            return "Error: LLM not loaded."
            
        # FORCE the model to use the context by making it the ONLY information available
        full_content = (
            f"Context from web search:\n"
            f"{search_context}\n\n"
            f"User question: {user_text}\n\n"
            f"Answer the question using ONLY the context above. "
            f"Be brief (1-2 sentences). "
            f"If the context doesn't have the answer, say 'The search results don't contain that information.'"
        )
        
        messages = [{"role": "user", "content": full_content}]
          
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.3,  # Lower temperature for more factual
                stop=["<end_of_turn>", "User:", "<start_of_turn>"]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error during search inference: {e}")
            return "I couldn't process the search results."

if __name__ == "__main__":
    bot = PocketLLM()
    if bot.llm:
        print("Test Response:", bot.generate_response("Hello, who are you?"))
