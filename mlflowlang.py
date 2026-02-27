import mlflow
from mlflow.genai.optimize.optimizers import MetaPromptOptimizer
from langchain_ollama import ChatOllama

# --- CONFIGURATION ---
mlflow.set_tracking_uri("sqlite:///fwdslash_prompts.db") 

class FwdSlashZeroShot:
    def __init__(self, user_id: str, agent_id: str):
        self.user_id = user_id
        self.agent_id = agent_id
        self.prompt_name = f"fwdslash_{user_id}_{agent_id}_zero"
        mlflow.set_experiment(f"ZeroShot_{user_id}")

    def register_initial_prompt(self, template: str):
        try:
            return mlflow.genai.register_prompt(name=self.prompt_name, template=template)
        except Exception:
            return mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@latest")

    def run_zero_shot_optimization(self, instructions: str):
        """Optimizes the prompt based ONLY on guidelines, no dataset needed."""
        current_prompt = mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@latest")

        print(f"\n--- Refining Prompt via Meta-Guidelines ---")
        
        # We pass empty lists for train_data and scorers
        result = mlflow.genai.optimize_prompts(
            predict_fn=lambda **kwargs: "", # Dummy function since no evaluation happens
            train_data=[], 
            prompt_uris=[current_prompt.uri],
            optimizer=MetaPromptOptimizer(
                reflection_model="ollama:/llama3.1",
                guidelines=instructions
            ),
            scorers=[] 
        )
        print("--- Optimization Complete ---\n")
        return result

    def get_latest_response(self, **kwargs):
        """Standard execution using the newly refined prompt."""
        prompt_obj = mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@latest")
        llm = ChatOllama(model="llama3.1", temperature=0)
        return llm.invoke(prompt_obj.format(**kwargs)).content

# --- EXECUTION ---

if __name__ == "__main__":
    # 1. Setup
    fwd_zero = FwdSlashZeroShot(user_id="user_001", agent_id="support_bot")
    
    # 2. Start with a very basic prompt
    fwd_zero.register_initial_prompt("Help the user: {{customer_query}}")

    # 3. Optimize using ONLY Guidelines
    # This is where you define the 'rules' for FwdSlash
    fwd_guidelines = """
    1. Act as a professional customer support agent for FwdSlash AI.
    2. Keep responses under 3 sentences.
    3. Always include a polite greeting.
    4. If you don't know the answer, ask for their email address to follow up.
    """
    
    fwd_zero.run_zero_shot_optimization(instructions=fwd_guidelines)

    # 4. See the results
    print("Testing the Zero-Shot Optimized prompt...")
    response = fwd_zero.get_latest_response(customer_query="How do I integrate Zapier?")
    print(f"\nFinal Response:\n{response}")