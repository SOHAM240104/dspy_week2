import os
import dspy
import bs4
import mlflow
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dspy.teleprompt import BootstrapFewShot

os.environ["USER_AGENT"] = "LocalRAG/1.0"

# --- 1. CONFIGURE MODELS ---
STUDENT_MODEL = 'ollama_chat/llama3.2'
TEACHER_MODEL = 'ollama_chat/llama3.1:8b'

student_lm = dspy.LM(STUDENT_MODEL, api_base='http://localhost:11434', cache=True)
teacher_lm = dspy.LM(TEACHER_MODEL, api_base='http://localhost:11434', cache=True)
dspy.configure(lm=student_lm)

# --- 2. DATA INGESTION ---
print("\n[INIT] 1. Loading and indexing the blog...")
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_db" 
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- FLOW VISUALIZATION 1 ---
def search(query: str):
    print(f"\n🔗 --- FLOW STEP 1: LangChain Retrieval ---")
    print(f"🔍 [LangChain] Searching local ChromaDB for: '{query}'")
    results = retriever.invoke(query)
    print(f"📄 [LangChain] Retrieved {len(results)} relevant chunks.")
    return [r.page_content for r in results]

# --- 3. AUTO-GENERATE TRAINING DATA ---
print("\n[INIT] 2. Teacher LLM is auto-generating the training dataset...")
trainset = []
for chunk in all_splits[5:10]: 
    prompt = f"""Read this text: {chunk.page_content}
    Write one specific question that can be answered by this text, and provide a short, accurate answer.
    Format exactly like this:
    Q: [Your Question]
    A: [Your Answer]
    """
    response = teacher_lm(prompt)[0]
    try:
        q = response.split("Q:")[1].split("A:")[0].strip()
        a = response.split("A:")[1].strip()
        trainset.append(dspy.Example(question=q, answer=a).with_inputs('question'))
    except Exception as e:
        continue

# --- SHOW THE GENERATED QUESTIONS ---
print("\n📝 --- SYNTHETIC DATASET GENERATED ---")
for i, ex in enumerate(trainset):
    print(f"Example {i+1} | Q: {ex.question}")
    print(f"          | A: {ex.answer}")

# --- 4. THE DSPy MODULE ---
class RAGSignature(dspy.Signature):
    """Answer technical questions based on the provided context."""
    context = dspy.InputField(desc="Relevant snippets from the blog")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A detailed yet concise technical answer")

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        context = search(question)
        print(f"\n🧠 --- FLOW STEP 2: DSPy Orchestration ---")
        print(f"⚙️ [DSPy] Assembling prompt using {len(context)} LangChain chunks + Optimized Examples...")
        prediction = self.respond(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer, reasoning=prediction.reasoning)

# --- 5. THE TEACHER JUDGE METRIC ---
def teacher_judge(example, pred, trace=None):
    judge_prompt = f"""You are a fair but strict teacher grading a student.
    Context: {pred.context}
    Question: {example.question}
    Student's Answer: {pred.answer}
    Does the Student's Answer correctly answer the Question based on the Context? 
    Respond with exactly 'YES' or 'NO' and nothing else."""
    
    response = teacher_lm(judge_prompt)[0].strip().upper()
    result = "✅ PASS" if "YES" in response else "❌ FAIL"
    print(f"\n   👨‍🏫 [Teacher Judge] Grading Student's answer: '{pred.answer[:60]}...' -> {result}")
    return "YES" in response

# ==========================================
# --- 6. MLFLOW TRACKING & OPTIMIZATION ---
# ==========================================
mlflow.set_experiment("DSPy_RAG_Optimization")

with mlflow.start_run(run_name="Llama3.2_Student_Llama3.1_Teacher"):
    mlflow.log_param("student_model", STUDENT_MODEL)
    mlflow.log_param("teacher_model", TEACHER_MODEL)
    mlflow.log_param("training_examples", len(trainset))
    
    print(f"\n[INIT] 3. Starting DSPy Optimization using Teacher Judge on {len(trainset)} examples...")
    optimizer = BootstrapFewShot(metric=teacher_judge)
    optimized_rag = optimizer.compile(RAG(), trainset=trainset)
    
    model_file = "auto_agent_v1.json"
    optimized_rag.save(model_file)
    mlflow.log_artifact(model_file)
    
    num_successful_traces = len(optimized_rag.predictors()[0].demos)
    mlflow.log_metric("successful_traces", num_successful_traces)
    print(f"\n✅ Saved optimized model with {num_successful_traces} traces to MLflow!\n")

    # --- 7. EXECUTION ---
    print("="*50)
    query = "What are the limitations of using LLMs as planning modules?"
    print(f"Test Question: {query}")
    
    response = optimized_rag(question=query)
    
    print(f"\n🧠 --- REASONING ---\n{response.reasoning}")
    print(f"\n✅ --- FINAL ANSWER ---\n{response.answer}")
    
    mlflow.log_param("test_query", query)
    mlflow.log_text(response.answer, "final_answer.txt")
    
    # --- THE BUG FIX ---
    # Safely extract the prompt depending on how DSPy/LiteLLM formatted it
    last_history = student_lm.history[-1]
    last_prompt_text = last_history.get('prompt') or str(last_history.get('messages', 'Prompt history unavailable'))
    mlflow.log_text(last_prompt_text, "optimized_prompt_sent_to_llm.txt")

# --- 8. SHOW THE PROMPT IN TERMINAL ---
print("\n" + "="*50)
print("👀 --- THE OPTIMIZED PROMPT ---")
print("Here is the massive, optimized prompt DSPy secretly sent to Llama 3.2 to get that answer:\n")
student_lm.inspect_history(n=1)