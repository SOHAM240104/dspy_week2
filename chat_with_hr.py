import dspy
import warnings
import os
from sentence_transformers import SentenceTransformer
from langchain_classic.memory import ConversationBufferWindowMemory

# 1. SETUP & CONFIGURATION
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuring DSPy to use your local Ollama instance
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='local')
dspy.configure(lm=lm)

# 2. THE SEARCH INDEX (Retriever)
corpus = [
    "Employee Handbook v1.0. Purpose: sets expectations for all. Updates live in Knowledge Base.",
    "EEO Policy: We prohibit discrimination. Report concerns to manager or HR.",
    "Harassment: includes unwelcome verbal/physical conduct. HR investigates impartially.",
    "Classifications: full-time, part-time, temporary, intern, contractor. Payroll is biweekly.",
    "Standard hours: 9:00–17:30. Core collab: 10:00–16:00. Home-office stipend: $500.",
    "Growth: L1–L7 leveling framework. Annual review in Q1. Learning stipend: $1,000/year.",
    "Benefits: Medical/Dental/Vision start 1st of month post-hire. 401(k) match up to 4%.",
    "Time Off: Accrue 1.5 days PTO/month (18/year). Carry over 5 days. Parental leave: 16 weeks.",
    "Equipment: Choose MacBook Pro or Dell XPS. MFA required. Loss must be reported in 24h.",
    "Expenses: Under $75 no receipt needed. No alcohol reimbursement. Submit within 30 days."
]

print("Setting up local retrieval (CPU)...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embedder = dspy.Embedder(st_model.encode, batch_size=64)
search = dspy.retrievers.Embeddings(corpus=corpus, embedder=embedder, k=2)

# 3. DEFINE MODULE SIGNATURE
class HRAnswer(dspy.Signature):
    """Answer HR questions using retrieved context. Use step-by-step reasoning."""
    history: str = dspy.InputField(desc="Previous conversation history")
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="relevant handbook snippets")
    answer: str = dspy.OutputField()

class OptimizedMiniHR(dspy.Module):
    def __init__(self):
        super().__init__()
        # MATCHING KEY: 'self.answer' matches "answer.predict" in your JSON
        self.answer = dspy.ChainOfThought(HRAnswer)

    def forward(self, history, question):
        # Perform retrieval
        ctx_passages = search(question).passages
        context = "\n\n".join(f"- {p}" for p in ctx_passages)
        
        # Pass all fields to the optimized predictor
        return self.answer(history=history, question=question, context=context)

# 4. INITIALIZE & LOAD OPTIMIZATION
rag_bot = OptimizedMiniHR()
try:
    # This specifically looks for "answer.predict" inside the JSON
    rag_bot.load("hr_rag_llama_optimized.json")
    print("✅ Successfully loaded MIPROv2 Optimized Prompt.")
except Exception as e:
    print(f"⚠️ Load failed: {e}. Running baseline (unoptimized).")

# Setup LangChain Memory (Window of 5 turns)
lc_memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history")

# 5. CHAT EXECUTION LOOP
def run_chat():
    print("\n--- 🦙 OLLAMA HR BOT ONLINE (Optimized + Memory) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("🧑 You: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # Clear history to see only the current turn's prompt
        lm.history.clear()

        # A. Pull chat history from LangChain
        history_str = lc_memory.load_memory_variables({})["chat_history"]

        # B. Run the DSPy Module
        response = rag_bot(history=history_str, question=user_query)

        # C. Update LangChain Memory
        lc_memory.save_context({"input": user_query}, {"output": response.answer})

        # D. Output result
        print(f"\n🤖 Bot: {response.answer}")

        # E. THE REVEAL: Show the improvised, optimized prompt
        print("\n" + "="*60)
        print("🛠️  IMPROVISED PROMPT (Optimized Instructions + Demos):")
        lm.inspect_history(n=1)
        print("="*60 + "\n")

if __name__ == "__main__":
    run_chat()