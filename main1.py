import dspy
import mlflow
import json
from typing import List


# =====================================================
# 🔐 Setup
# =====================================================

MY_GROQ_API_KEY = "YOUR_KEY_HERE"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DSPy_Adapter_Capabilities")
mlflow.dspy.autolog()


lm = dspy.LM(
    model="groq/llama-3.3-70b-versatile",
    api_key=MY_GROQ_API_KEY
)


# =====================================================
# 🧠 Tools
# =====================================================

def verify_scam(query: str) -> str:
    """Checks Indian scam patterns."""
    return f"Scam DB: '{query}' matches known KBC lottery fraud (confidence 0.94)."


def extract_entities(query: str) -> str:
    """Extracts entities from a suspicious message."""
    return "Entities: [KBC, Lottery, 1 Crore, URL]"


available_tools = [
    dspy.Tool(verify_scam),
    dspy.Tool(extract_entities)
]


# =====================================================
# 🧩 Signature
# =====================================================

class ToolSignature(dspy.Signature):
    """
    Decide which tools to call and emit native tool calls if possible.
    """
    query: str = dspy.InputField()
    tools: List[dspy.Tool] = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()
    rationale: str = dspy.OutputField(desc="Optional reasoning if adapter allows")


# =====================================================
# 🔬 Adapter Experiments
# =====================================================

adapters = {
    "native_function_calling": dspy.ChatAdapter(use_native_function_calling=True),
    "react_style": dspy.ChatAdapter(use_native_function_calling=False),
    "no_adapter": None
}


QUERY = "Verify this: 'You won a lottery of 1 Cr from KBC, click here.'"


for adapter_name, adapter in adapters.items():

    print(f"\n\n==============================")
    print(f"🧪 Adapter: {adapter_name}")
    print(f"==============================")

    dspy.configure(lm=lm, adapter=adapter)

    predictor = dspy.Predict(ToolSignature)

    with mlflow.start_run(run_name=f"adapter_{adapter_name}"):

        prediction = predictor(
            query=QUERY,
            tools=available_tools
        )

        # -------------------------------------------------
        # 📝 Log Raw Outputs
        # -------------------------------------------------
        mlflow.log_param("adapter", adapter_name)

        if hasattr(prediction, "rationale"):
            mlflow.log_text(prediction.rationale or "", "rationale.txt")

        if prediction.tool_calls:
            mlflow.log_text(
                json.dumps(prediction.tool_calls, default=str, indent=2),
                "tool_calls.json"
            )

        # -------------------------------------------------
        # 🔁 Manual Execution Loop
        # -------------------------------------------------
        if prediction.tool_calls:
            for call in prediction.tool_calls:
                name = getattr(call, 'name', call[0])
                args = getattr(call, 'args', call[1])

                print(f"🛠️ Tool Called → {name}")
                print(f"📦 Args → {args}")

                if name == "verify_scam":
                    out = verify_scam(**args)
                elif name == "extract_entities":
                    out = extract_entities(**args)
                else:
                    out = "Unknown tool"

                print(f"✅ Output → {out}")

        else:
            print("❌ No tool calls generated")


print("\n🚀 Open MLflow UI and compare runs side-by-side")
print("👉 Look at:")
print("   • tool_calls.json")
print("   • rationale.txt")
print("   • adapter param")