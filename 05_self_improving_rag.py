import dspy
from sentence_transformers import SentenceTransformer

# 1. Connect to Local Llama 3.1 (Infinite Quota!)
lm = dspy.LM(
    'ollama_chat/llama3.1', 
    api_base='http://localhost:11434', 
    api_key='local'
)
dspy.configure(lm=lm)

# 2. HR Handbook Snippets
corpus: list[str] = [
    "Employee Handbook v1.0. Purpose: sets expectations for all. Updates live in Knowledge Base. Questions? Email hr@company.example.",
    "EEO Policy: We prohibit discrimination. Report concerns to manager or HR. Retaliation is forbidden.",
    "Harassment: includes unwelcome verbal/physical conduct. HR investigates impartially and maintains confidentiality.",
    "Classifications: full-time, part-time, temporary, intern, contractor. Payroll is biweekly on Fridays.",
    "Standard hours: 9:00–17:30. Core collab: 10:00–16:00. Home-office stipend: $500 for new hires.",
    "Growth: L1–L7 leveling framework. Annual review in Q1. Learning stipend: $1,000/year.",
    "Benefits: Medical/Dental/Vision start 1st of month post-hire. 401(k) match up to 4%.",
    "Time Off: Accrue 1.5 days PTO/month (18/year). Carry over 5 days. Parental leave: 16 weeks.",
    "Equipment: Choose MacBook Pro or Dell XPS. MFA required. Loss must be reported in 24h.",
    "Expenses: Under $75 no receipt needed. No alcohol reimbursement. Submit within 30 days.",
    "Privacy: Handle personal data per law. Offboarding: Return all assets and deprovisioning."
]

# 3. Retrieval (CPU)
print("Setting up local retrieval...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embedder = dspy.Embedder(st_model.encode, batch_size=64)
search = dspy.retrievers.Embeddings(corpus=corpus, embedder=embedder, k=3)

# 4. Program Definition
class HRAnswer(dspy.Signature):
    """Answer HR questions using retrieved context. Use step-by-step reasoning but keep the final answer short."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="relevant handbook snippets")
    answer: str = dspy.OutputField()

class MiniHR(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(HRAnswer)

    def forward(self, question: str):
        ctx_passages = search(question).passages
        context = "\n\n".join(f"- {p}" for p in ctx_passages)
        return self.answer(question=question, context=context)

rag_bot = MiniHR()

# 5. Training Data
raw_eval = [
    ("How many PTO days do we get each year?", "18"),
    ("Do I need a receipt for a $50 lunch?", "No"),
    ("Is alcohol reimbursable?", "No"),
    ("What laptops can I choose?", "MacBook Pro and Dell XPS"),
]
trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in raw_eval]

def semantic_metric(example, pred, trace=None):
    gold = example.answer.lower()
    predicted = pred.answer.lower()
    return int(gold in predicted or predicted in gold)

# 6. MIPROv2 Optimization (Unleashed)
# auto="light" will work now because we won't get rate limited.
optimizer = dspy.MIPROv2(
    metric=semantic_metric, 
    auto="light", 
    num_threads=1, # Keep at 1 so your Mac doesn't freeze up
    verbose=True
)

# 7. Execution
def main():
    print("\n--- Running Unoptimized Baseline ---")
    for ex in trainset:
        print(f"Q: {ex.question} -> Ans: {rag_bot(question=ex.question).answer}")

    print("\n🚀 Compiling Optimized Bot with MIPROv2...")
    # This might take a few minutes as Llama grinds through the logic
    optimized_bot = optimizer.compile(rag_bot, trainset=trainset, valset=trainset)

    print("\n--- Optimized Result ---")
    for ex in trainset:
        print(f"Q: {ex.question} -> Ans: {optimized_bot(question=ex.question).answer}")

    optimized_bot.save("hr_rag_llama_optimized.json")
    print("\n💾 Success! Program saved to hr_rag_llama_optimized.json")

if __name__ == "__main__":
    main()