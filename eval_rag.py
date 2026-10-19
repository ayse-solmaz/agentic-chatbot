from agentic_chatbot_rag_backend import graph
from eval_dataset import DATASET_NAME
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import evaluate
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

# compile without the SQLite checkpointer so eval runs
# don't show up as conversations in the Streamlit sidebar
eval_chatbot = graph.compile()


# ========================= Run function =========================

def run_chatbot(inputs: dict) -> dict:
    result = eval_chatbot.invoke(
        {"messages": [HumanMessage(content=inputs["question"])]}
    )
    messages = result["messages"]

    # names of all tools the model called, in order
    tools_called = [
        tool_call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]

    # PDF chunks returned by rag_tool
    retrieved_context = "\n\n".join(
        message.content
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "rag_tool"
    )

    return {
        "answer": messages[-1].content,
        "tools_called": tools_called,
        "retrieved_context": retrieved_context,
    }


# ========================= Evaluators =========================

def get_outputs(obj):
    # local evaluate() passes objects, uploaded evaluators get dicts
    return obj.outputs if hasattr(obj, "outputs") else obj.get("outputs", {}) or {}


class CorrectnessGrade(TypedDict):
    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_correct: Annotated[bool, ..., "True if the answer matches the reference answer"]


class GroundednessGrade(TypedDict):
    reasoning: Annotated[str, ..., "Explain your reasoning"]
    is_grounded: Annotated[bool, ..., "True if every claim in the answer is supported by the context"]


judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
correctness_judge = judge.with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
groundedness_judge = judge.with_structured_output(GroundednessGrade, method="json_schema", strict=True)


def correctness(run, example):
    """Does the answer contain the same facts as the reference answer?"""
    run_outputs = get_outputs(run)
    example_outputs = get_outputs(example)

    grade = correctness_judge.invoke([
        {
            "role": "system",
            "content": (
                "You grade answers from a chatbot about a research paper. "
                "Compare the chatbot answer with the reference answer. "
                "The answer is correct if it contains the key facts and numbers of the reference "
                "and does not contradict it. Extra details are fine. "
                "The answer may be in a different language than the reference."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {example.inputs['question']}\n\n"
                f"Reference answer: {example_outputs['answer']}\n\n"
                f"Chatbot answer: {run_outputs['answer']}"
            ),
        },
    ])

    return {"score": 1 if grade["is_correct"] else 0, "comment": grade["reasoning"]}


def groundedness(run, example):
    """Is the answer supported by the PDF chunks rag_tool retrieved?"""
    run_outputs = get_outputs(run)
    context = run_outputs.get("retrieved_context", "")

    if not context:
        return {"score": 0, "comment": "rag_tool was not called, so the answer is not grounded in the PDF."}

    grade = groundedness_judge.invoke([
        {
            "role": "system",
            "content": (
                "You check whether a chatbot answer is grounded in the retrieved document context. "
                "The answer is grounded if every factual claim in it can be found in the context. "
                "Claims that are not in the context count as hallucinations."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nChatbot answer: {run_outputs['answer']}",
        },
    ])

    return {"score": 1 if grade["is_grounded"] else 0, "comment": grade["reasoning"]}


def used_rag_tool(run, example):
    """Did the model call the expected tool?"""
    run_outputs = get_outputs(run)
    example_outputs = get_outputs(example)

    expected = example_outputs.get("expected_tool", "rag_tool")
    called = run_outputs.get("tools_called", [])

    return {
        "score": 1 if expected in called else 0,
        "comment": f"Expected {expected}, called {called}",
    }


# ========================= Run evaluation =========================

if __name__ == "__main__":
    results = evaluate(
        run_chatbot,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness, used_rag_tool],
        experiment_prefix="rag-baseline",
        max_concurrency=4,
    )
    print(results.to_pandas()[[
        "inputs.question",
        "feedback.correctness",
        "feedback.groundedness",
        "feedback.used_rag_tool",
    ]])
