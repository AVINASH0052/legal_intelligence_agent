import os
import json
from pathlib import Path
from datetime import datetime
import sys

from rag_agent.retriever import Retriever
from rag_agent.agent import CaseFile, run_agent
from rag_agent.llm import NvidiaChatClient
from rag_agent.agent import submit_feedback

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "corpus.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case_2025 = CaseFile(
    title="2025 Policy: Mandatory Biometrics for Public Services",
    facts=(
        "Government policy requires all citizens to submit biometric data to access public services. "
        "A petitioner argues this violates privacy under Article 21."
    ),
    issues_hint=[
        "Is privacy (Art. 21) infringed?",
        "Is there a valid enabling law?",
        "Does the policy pass proportionality (suitability, necessity, balancing)?",
        "Are adequate safeguards in place?",
        "Are less intrusive alternatives viable?",
    ],
)

def main():
    # Prepare retriever
    docs = Retriever.load_docs(str(DATA))
    retriever = Retriever(docs)

    # NVIDIA client
    client = NvidiaChatClient()

    # UI-like controls via env vars
    try:
        temp = float(os.environ.get("LIA_TEMP", "0.35"))
    except ValueError:
        temp = 0.35
    try:
        max_toks = int(os.environ.get("LIA_MAX_TOKENS", "1700"))
    except ValueError:
        max_toks = 1700
    style_pref = os.environ.get("LIA_STYLE")  # "bullets" | "prose" | None

    # Run agent
    result = run_agent(retriever, case_2025, client, temperature=temp, max_tokens=max_toks, style_preference=style_pref)

    try:
        # Print concise console view
        print("Issues:", ", ".join(result["issues"]))
        print("\nPlan:")
        for i, s in enumerate(result["plan"], 1):
            print(f"{i}. {s}")

        print("\nTop Precedents (scored):")
        for r in result["retrieved"][:4]:
            print(f"- {r['title']} ({r['year']}): score={r['score']:.2f}")

        print("\nGrounding (claims -> citations):")
        for c, cites in result["grounding"]:
            print(f"- {c}\n  cites: {', '.join(cites) if cites else '—'}")
        if result.get("grounding_snippets"):
            print("\nGrounding snippets:")
            for item in result["grounding_snippets"]:
                print(f"- Claim: {item['claim']}")
                for ev in item.get("evidence", []):
                    print(f"  • {ev['source']}: {ev['snippet']}")

        print("\nDraft:\n")
        print(result["draft"])

        print("\nSuggested verdict confidence:", result["confidence"]) 
        sys.stdout.flush()

        # Simple interactive feedback (skips when not a TTY)
        if sys.stdin.isatty():
            print("\nWas this helpful? [Y/n] ", end="", flush=True)
            ans = sys.stdin.readline().strip().lower() or "y"
            thumbs_up = ans.startswith("y")
            notes = ""
            if thumbs_up:
                print("Any brief notes to keep improving? (Enter to skip): ", end="", flush=True)
                notes = sys.stdin.readline().strip()
            fb = submit_feedback(thumbs_up, used_doc_ids=result.get("used_doc_ids", []), notes=notes)
            print("Feedback recorded.")
    except BrokenPipeError:
        try:
            sys.stdout.flush()
        except Exception:
            pass
    finally:
        # Save a JSON and a txt snapshot
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        (OUTPUT_DIR / f"result_{ts}.json").write_text(json.dumps(result, indent=2))
        (OUTPUT_DIR / f"result_{ts}.txt").write_text(result["draft"]) 

if __name__ == "__main__":
    main()
