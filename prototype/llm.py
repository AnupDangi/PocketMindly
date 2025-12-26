from llama_cpp import Llama
import os
import sys

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
        
        # Initial system prompt
        self.system_prompt = "You are PocketMindly, an offline voice assistant. You answer in short, concise sentences. You are helpful and calm."

    def generate_response(self, user_text):
        """
        Generates a response from the LLM.
        """
        if not self.llm:
            return "Error: LLM not loaded."

        # Construct prompt for Gemma Instruct
        # Format: <start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n
        
        messages = [
            {"role": "user", "content": self.system_prompt + "\n\nUser: " + user_text}
        ]
        
        # Simplified prompting for direct completion if simple chat method isn't used
        # But llama-cpp-python has create_chat_completion which handles chat templates if metadata exists
        # Let's try raw completion with Gemma formatting to be safe or use high level API
        
        try:
             # Gemma 2B-IT does not support 'system' role in some templates.
            # We merge it into the user prompt.
            full_prompt = f"{self.system_prompt}\n\nUser: {user_text}"
            
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500,
                temperature=0.7,
                stop=["<end_of_turn>"]
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error during inference: {e}")
            return "I'm having trouble thinking."

if __name__ == "__main__":
    bot = PocketLLM()
    if bot.llm:
        print("Test Response:", bot.generate_response("Hello, who are you?"))
