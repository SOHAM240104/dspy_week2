import dspy
import re

# ---------------------------------------------------------
# 1. CONNECT TO LOCAL LLAMA (VS CODE + OLLAMA)
# ---------------------------------------------------------
lm = dspy.LM(
    'ollama_chat/llama3.1:8b', 
    api_base='http://localhost:11434', 
    api_key='local'
)
dspy.configure(lm=lm)

# ---------------------------------------------------------
# 2. DEFINE PYTHON TOOLS
# ---------------------------------------------------------
def get_exchange_rate(currency_code: str) -> float:
    """Returns the USD conversion rate for a given currency code."""
    # Mock data for demonstration
    fx_rates = {"USD": 1.0, "EUR": 1.07, "GBP": 1.26, "INR": 0.012}
    rate = fx_rates.get(currency_code.upper(), 0.0)
    print(f"DEBUG: FX Tool called for {currency_code} -> {rate}")
    return rate

def calculate(expression: str) -> float:
    """Runs a safe math calculation."""
    # Security: only allow basic math characters
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError(f"Dangerous expression detected: {expression}")
    result = eval(expression)
    print(f"DEBUG: Calc Tool called for {expression} -> {result}")
    return result

# ---------------------------------------------------------
# 3. WRAP TOOLS & SETUP REACT
# ---------------------------------------------------------
# Wrapping functions as dspy.Tools tells the AI how to use them
tools = [
    dspy.Tool(get_exchange_rate, name="FX_Rate", desc="Get conversion to USD"),
    dspy.Tool(calculate, name="Math_Calc", desc="Solve math expressions")
]

# ReAct (Reason + Act) is the best pattern for tool-calling
agent = dspy.ReAct("question -> answer", tools=tools)

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
def main():
    query = "I spent 120 EUR on dinner. What is that in USD? Is it above our $75 budget?"
    
    print(f"🚀 Running Query: {query}\n")
    
    # max_iters=5 keeps the local model from looping forever if it gets confused
    prediction = agent(question=query, max_iters=5)

    print("\n" + "="*30)
    print("FINAL ANSWER:", prediction.answer)
    print("="*30)

    # See the Thought -> Action -> Observation steps
    print("\n🧠 STEP-BY-STEP TRAJECTORY:")
    for step in prediction.trajectory:
        print(f"➜ {step}")

if __name__ == "__main__":
    main()