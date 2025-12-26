from typing import List, Dict

class PromptManager:
    """
    Manages the construction of prompts for the LLM, including:
    - System Instructions (Persona)
    - Few-Shot Examples (Tone Guide)
    """

    SYSTEM_INSTRUCTION = """
You are PocketMindly, a calm, intelligent, offline-first AI assistant.

Your goals:
- Answer questions clearly and briefly when possible.
- Reason normally about abstract, philosophical, or opinion-based questions.
- Use common sense and general knowledge freely.

Web search rules:
- Only request web search when the question requires:
  1) very recent information (news, prices, current events)
  2) precise factual lookup you are unsure about
- Do NOT request web search for:
  - philosophical questions
  - opinions or beliefs
  - hypothetical questions
  - general knowledge you can explain

If web search is needed, respond EXACTLY in this format:
SEARCH_REQUEST: <concise search query>

Otherwise, answer normally.

Be brief. Be natural. Do not mention tools unless requesting search.
"""

    # Few-Shot Examples: (User Input, Ideal AI Output)
    FEW_SHOT_EXAMPLES = [
    ("What is the capital of France?", "Paris."),
    ("What are today’s bitcoin prices?", "SEARCH_REQUEST: current bitcoin price"),
    ("Does consciousness come from the brain?", 
     "This is a debated topic. Many scientists believe consciousness emerges from brain activity, but there is no single accepted explanation.")
]

    def construct_messages(self, user_text: str) -> List[Dict[str, str]]:
        """
        Constructs the message list for the chat completion API.
        Merges System Instruction into the first User message for Gemma compatibility.
        Injects Few-Shot examples as history.
        """
        messages = []

        # 1. Start with the First User Message containing the System Instruction
        # We attach the first example to this to "prime" the model immediately.
        first_example_user, first_example_ai = self.FEW_SHOT_EXAMPLES[0]
        
        first_content = (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"User: {first_example_user}"
        )
        
        messages.append({"role": "user", "content": first_content})
        messages.append({"role": "assistant", "content": first_example_ai})

        # 2. Add remaining Few-Shot Examples as history
        for ex_user, ex_ai in self.FEW_SHOT_EXAMPLES[1:]:
            messages.append({"role": "user", "content": f"User: {ex_user}"})
            messages.append({"role": "assistant", "content": ex_ai})

        # 3. Append the Actual User Input
        messages.append({"role": "user", "content": f"User: {user_text}"})

        return messages
