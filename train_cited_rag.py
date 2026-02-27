import dspy
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# 1. SETUP LOCAL LLAMA
# -------------------------------------------------------
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='local')
dspy.configure(lm=lm)

# -------------------------------------------------------
# 2. LOAD 50-ROW DATASET
# -------------------------------------------------------
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Ensure answer is string to avoid float errors
    return [dspy.Example(question=row['question'], answer=str(row['answer'])).with_inputs("question") for _, row in df.iterrows()]

trainset = load_dataset('hr_data_50.csv')

# -------------------------------------------------------
# 3. RAG COMPONENTS (Retriever)
# -------------------------------------------------------
corpus = [
    "PTO: 18 days/year. Carry over 5 days.",
    "Stipend: $500 for new hires.",
    "Laptops: MacBook Pro or Dell XPS.",
    "Expenses: Under $75 no receipt needed. No alcohol reimbursement.",
    "Payday: Biweekly on Fridays."
]
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embedder = dspy.Embedder(st_model.encode)
retriever = dspy.retrievers.Embeddings(corpus=corpus, embedder=embedder, k=3)

# -------------------------------------------------------
# 4. SIGNATURE & MODULE
# -------------------------------------------------------
class CitedHRAnswer(dspy.Signature):
    """Answer HR questions using context. Provide a citation from the snippets used."""
    question = dspy.InputField()
    context = dspy.InputField(desc="relevant handbook snippets")
    answer = dspy.OutputField(desc="concise policy answer")
    citation = dspy.OutputField(desc="exact quote or snippet title used as source")

class CitedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(CitedHRAnswer)

    def forward(self, question):
        passages = retriever(question).passages
        context = "\n".join([f"SOURCE {i+1}: {p}" for i, p in enumerate(passages)])
        return self.respond(question=question, context=context)

# -------------------------------------------------------
# 5. TRACE-AWARE METRIC (THE FIX)
# -------------------------------------------------------
def citation_metric(gold, pred, trace=None):
    # 1. Extract the context from the trace if available
    # During optimization, DSPy provides a trace of the call.
    # The context was an input to the 'respond' component.
    context = ""
    if trace is not None:
        # We look for the 'context' in the inputs of the last called module
        context = trace[-1][1].get('context', "")
    
    # 2. Accuracy Check
    answer_correct = gold.answer.lower() in pred.answer.lower()
    
    # 3. Citation Check
    citation = getattr(pred, 'citation', "")
    has_citation = len(citation) > 5
    
    # 4. Grounding Check (Hallucination Guard)
    is_grounded = False
    if context and citation:
        is_grounded = citation.lower() in context.lower()
    
    # For debugging in console
    if not is_grounded and has_citation:
        print(f"DEBUG: Citation Hallucination detected for: {gold.question}")

    return int(answer_correct and has_citation and is_grounded)

# -------------------------------------------------------
# 6. RUN OPTIMIZER
# -------------------------------------------------------
# auto="light" is perfect for Llama 3.1 local runs
optimizer = dspy.MIPROv2(metric=citation_metric, auto="light", num_threads=1)

print("\n🚀 Starting Cited RAG Optimization...")
print("This will take a few minutes as Llama learns to provide evidence...")

# Using a subset of trainset for faster light run if needed, but 50 is fine.
optimized_cited_rag = optimizer.compile(CitedRAG(), trainset=trainset)

# 7. SAVE THE BRAIN
optimized_cited_rag.save("hr_cited_rag_optimized.json")
print("\n✅ Success! Compiled 'Brain' saved to hr_cited_rag_optimized.json")