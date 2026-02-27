import dspy
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# 1. LOCAL LLM (Ollama)
# -------------------------------------------------------

lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    api_key="local"
)
dspy.configure(lm=lm)

# -------------------------------------------------------
# 2. HR HANDBOOK CORPUS (YOUR DATA)
# -------------------------------------------------------

corpus = [
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

# -------------------------------------------------------
# 3. RETRIEVAL (EMBEDDINGS)
# -------------------------------------------------------

st_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

embedder = dspy.Embedder(st_model.encode)

retriever = dspy.retrievers.Embeddings(
    corpus=corpus,
    embedder=embedder,
    k=3
)

# -------------------------------------------------------
# 4. SIGNATURE
# -------------------------------------------------------

class HRAnswer(dspy.Signature):
    """
    Answer HR questions using retrieved context.
    Focus on policy rules and provide a short final answer.
    """
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="relevant handbook snippets")
    answer: str = dspy.OutputField()

# -------------------------------------------------------
# 5. RAG MODULE
# -------------------------------------------------------

class HRRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(HRAnswer)

    def forward(self, question: str):
        passages = retriever(question).passages
        context = "\n".join(f"- {p}" for p in passages)
        return self.answer(question=question, context=context)

rag_program = HRRAG()

# -------------------------------------------------------
# 6. TRAINING EXAMPLES (NO CSV)
# -------------------------------------------------------

trainset = [
    dspy.Example(
        question="How many PTO days do we get each year?",
        answer="18"
    ).with_inputs("question"),

    dspy.Example(
        question="Is alcohol reimbursable?",
        answer="No"
    ).with_inputs("question"),

    dspy.Example(
        question="Do I need a receipt for a $50 lunch?",
        answer="No"
    ).with_inputs("question"),

    dspy.Example(
        question="What laptops can I choose?",
        answer="MacBook Pro and Dell XPS"
    ).with_inputs("question"),
]

# -------------------------------------------------------
# 7. METRIC
# -------------------------------------------------------

def simple_metric(example, pred, trace=None):
    gold = example.answer.lower()
    predicted = pred.answer.lower()
    return int(gold in predicted or predicted in gold)

# -------------------------------------------------------
# 8. GEPA OPTIMIZER (SELF-IMPROVING PROMPT)
# -------------------------------------------------------

optimizer = dspy.GEPA(
    metric=simple_metric,
    verbose=True
)

print("\n🚀 Running GEPA self-improvement...\n")

optimized_rag = optimizer.compile(
    rag_program,
    trainset=trainset
)

optimized_rag.save("gepa_hr_rag_optimized.json")

print("\n✅ Optimization complete.")

# -------------------------------------------------------
# 9. INTERACTIVE TEST
# -------------------------------------------------------

print("\n--- HR SELF-IMPROVING RAG BOT ---")

while True:
    question = input("\nAsk HR (or type exit): ")
    if question.lower() == "exit":
        break

    result = optimized_rag(question=question)
    print("Answer:", result.answer)