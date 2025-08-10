import os
from pathlib import Path
import streamlit as st
from rag_agent.retriever import Retriever
from rag_agent.agent import CaseFile, run_agent, submit_feedback
from rag_agent.llm import NvidiaChatClient

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "corpus.json"

@st.cache_resource
def get_retriever():
    docs = Retriever.load_docs(str(DATA))
    return Retriever(docs)

@st.cache_resource
def get_client():
    return NvidiaChatClient()

st.set_page_config(page_title="Legal Intelligence Agent", layout="wide")
st.title("Legal Intelligence Agentic System (RAG + NVIDIA LLM)")

with st.sidebar:
    st.markdown("### API Configuration")
    st.write("Set NVIDIA_API_KEY in your environment before running.")
    model = os.environ.get("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    st.code(f"Model: {model}")
    st.markdown("### Generation Controls")
    temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_toks = st.slider("Max tokens", 400, 3000, 1600, 50)

st.markdown("Enter case title and facts. The agent retrieves precedents, drafts arguments, and suggests an outcome.")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Case Title", "Biometric Authentication Mandate for Ration Cards")
with col2:
    facts = st.text_area("Facts", "State mandates fingerprints for ration disbursal; elderly and manual laborers face failures. Petitioners argue necessity and less intrusive alternatives.", height=120)

issues_hint = [
    "Is privacy (Art. 21) infringed?",
    "Is there a valid enabling law?",
    "Does the policy pass proportionality (suitability, necessity, balancing)?",
    "Are adequate safeguards in place?",
    "Are less intrusive alternatives viable?",
]

if st.button("Run Agent", type="primary"):
    retriever = get_retriever()
    client = get_client()
    case = CaseFile(title=title, facts=facts, issues_hint=issues_hint)
    with st.spinner("Running agent..."):
        try:
            result = run_agent(retriever, case, client, temperature=temp, max_tokens=int(max_toks))
        except Exception as e:
            import traceback
            st.error(f"LLM API error: {e}")
            st.caption("If this persists, check your NVIDIA API key, model, or try again later.")
            st.expander("Show traceback").write(traceback.format_exc())
            st.stop()
    st.subheader("Issues")
    st.write(", ".join(result["issues"]))

    st.subheader("Plan")
    for i, step in enumerate(result["plan"], 1):
        st.write(f"{i}. {step}")

    st.subheader("Top Precedents (scored)")
    for r in result["retrieved"][:4]:
        st.write(f"- {r['title']} ({r['year']}): score={r['score']:.2f}")

    st.subheader("Grounding (claims -> citations)")
    for c, cites in result["grounding"]:
        st.write(f"- {c}")
        st.caption(", ".join(cites) if cites else "—")

    if result.get("grounding_snippets"):
        st.subheader("Grounding snippets")
        for item in result["grounding_snippets"]:
            st.markdown(f"- Claim: {item['claim']}")
            for ev in item.get("evidence", []):
                st.caption(f"  • {ev['source']}: {ev['snippet']}")

    st.subheader("Draft")
    # Render as plain text (no markdown) in a read-only area, with download option
    st.text_area("Draft", result["draft"], height=480)
    st.download_button(
        label="Download draft (.txt)",
        data=result["draft"].encode("utf-8"),
        file_name="legal_brief.txt",
        mime="text/plain",
    )

    st.subheader("Suggested verdict confidence")
    st.write(result["confidence"]) 

    st.divider()
    st.markdown("### Feedback")
    helpful = st.toggle("This was helpful")
    notes = st.text_input("Notes (optional)", "")
    if st.button("Submit Feedback"):
        fb = submit_feedback(thumbs_up=helpful, used_doc_ids=result.get("used_doc_ids", []), notes=notes)
        st.success("Feedback recorded.")
        st.json(fb)

st.caption("Tip: adjust corpus in data/corpus.json to add more precedents.")
