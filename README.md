# Legal Intelligence Agentic System

A compact, modular pipeline for precedent-aware legal reasoning leveraging RAG + agentic workflow. Uses NVIDIA API via the OpenAI Python client.

## Features
- NumPy cosine vector search with sentence-transformer embeddings (no FAISS required).
- Composite precedent scoring: similarity + recency + court weight + issue overlap + small feedback boosts.
- Agentic flow: issue spotting → retrieval → plan → grounding → arguments → verdict.
- NVIDIA Inference API (OpenAI-compatible) with streaming-safe behavior.
- Simple feedback loop to nudge style and emphasis.
- Streamlit UI with temperature and max-token controls (passed through to drafting).

## Quickstart
1) Create a virtual environment (optional)
- macOS/Linux (zsh):
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Set NVIDIA API key and run CLI
```
export NVIDIA_API_KEY="<your_nvapi_key>"
python run.py
```

4) Launch Streamlit UI
```
export NVIDIA_API_KEY="<your_nvapi_key>"
streamlit run app.py
```

5) Outputs will print to console/UI and be saved under `outputs/`.

## Files
- `run.py` — main script: loads corpus, runs agent, prints/saves results.
- `app.py` — Streamlit UI: inputs, controls, outputs, feedback.
- `rag_agent/` — retriever, agent, NVIDIA LLM wrapper, feedback module.
- `data/corpus.json` — small seed corpus of paraphrased case notes.
- `docs/DESIGN.md` — architecture and component interactions.
- `outputs/` — saved sample responses.

## Troubleshooting
- Ensure `sentence-transformers` downloads models (first run may be slow).
- If piping output (e.g., `| head`), streaming is auto-disabled to avoid BrokenPipe.
- Set `NVIDIA_MODEL` to switch models. Default: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- Set `FEEDBACK_PATH` to change feedback JSON location.

## Sample Cases to Test the RAG System

You can use these sample case titles and facts to check retrieval, argument, and verdict generation:

**1. Blanket Internet Shutdown for Exams**
- Title: 2024 State-wide Internet Shutdown During Competitive Exams
- Facts: The State imposed a blanket internet shutdown across all districts for two full days to prevent exam leaks. Petitioners argue this violates freedom of speech and proportionality under Articles 19(1)(a) and 21.

**2. Mandatory Data Localization for Fintech**
- Title: 2025 RBI Circular: Full Data Localization for Fintech Wallets
- Facts: RBI mandates all fintech wallets to store all user data solely in India. Petitioners argue this restricts trade (Art. 19(1)(g)) and is disproportionate; State claims security and investigatory interests.

**3. Citywide CCTV Without Statute**
- Title: Citywide CCTV Grid Without Enabling Law
- Facts: Municipal body deployed facial recognition CCTVs across markets without a statute. Petitioners raise privacy (Art. 21), legality, and safeguards concerns.

**4. Biometric Mandate for Ration Cards**
- Title: Biometric Authentication Mandate for Ration Cards
- Facts: State mandates fingerprints for ration disbursal; elderly and manual laborers face failures. Petitioners argue necessity and less intrusive alternatives.

**5. Solar Farm Ban and Right to Livelihood**
- Title: 2025 Solar Farm Ban and Right to Livelihood
- Facts: State bans new solar farms on agricultural land, citing food security. Petitioners (farmers, energy firms) claim violation of right to livelihood (Art. 21) and trade (Art. 19(1)(g)); State cites agricultural output concerns.

## Live Demo and Example Output

- **Live Streamlit App:** [View the app](http://localhost:8503)  
  *(Replace with your public URL if deployed, e.g., Streamlit Community Cloud or HuggingFace Spaces)*

- **Example Output:**


