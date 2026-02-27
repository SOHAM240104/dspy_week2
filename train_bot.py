import dspy
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. SETUP
# We use one LM instance to act as both the 'Student' and the 'Coach'
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='local')
dspy.configure(lm=lm)

# 2. DATA LOADER
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return [dspy.Example(question=row['question'], answer=str(row['answer'])).with_inputs("question") for _, row in df.iterrows()]

trainset = load_dataset('hr_data_50.csv')

# 3. RAG COMPONENTS
corpus = ["Standard hours: 9:00-17:30.", "PTO: 18 days/year.", "Stipend: $500.", "Laptops: MacBook Pro or Dell XPS.", "Report loss in 24h."]
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embedder = dspy.Embedder(st_model.encode)
retriever = dspy.retrievers.Embeddings(corpus=corpus, embedder=embedder, k=3)

class HRAnswer(dspy.Signature):
    """Answer HR questions strictly using context. Be very concise."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()

class HRRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(HRAnswer)
    def forward(self, question):
        context = "\n".join(retriever(question).passages)
        return self.answer(question=question, context=context)

# 4. GEPA METRIC (The 5-Argument requirement)
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    return int(gold.answer.lower() in pred.answer.lower())

# 5. GEPA OPTIMIZER (Now with the 'Coach' / reflection_lm)
# This is the line that was causing the last error!
optimizer = dspy.GEPA(
    metric=gepa_metric, 
    auto="light",
    reflection_lm=lm  # We are telling GEPA to use Llama 3.1 as the Coach
)

print("\n🚀 Starting GEPA Evolution...")
print("Step 1: Running baseline eval.")
print("Step 2: Llama is reflecting on mistakes.")
print("Step 3: Mutating the prompt...")

# Run the compilation
optimized_rag = optimizer.compile(HRRAG(), trainset=trainset)

# 6. SAVE
optimized_rag.save("gepa_hr_rag_optimized.json")
print("\n✅ WE ARE DONE! 'gepa_hr_rag_optimized.json' is ready for chat_bot.py.")