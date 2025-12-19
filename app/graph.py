from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.utils import get_llm


class AgentState(TypedDict):
    query: str
    answer: str


def build_graph(vectorstore):
    llm = get_llm()

    def retrieve_node(state: AgentState):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(state["query"])

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
        Answer the question using ONLY the context below.
        If the answer is not present, say "I don't know".

        Context:
        {context}

        Question:
        {state['query']}
        """

        response = llm.invoke(prompt)
        return {"answer": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", END)

    return workflow.compile()