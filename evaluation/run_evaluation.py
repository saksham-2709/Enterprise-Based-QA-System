import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from faiss_index import load_index
from orchestration.pipeline import RAGPipeline


def run_evaluation():
    print("\n=== Running RAG Evaluation ===\n")

    index = load_index("faiss.index")
    data = json.load(open("meta.json", "r", encoding="utf-8"))

    pipeline = RAGPipeline(
        index=index,
        chunks=data["chunks"],
        metas=data["metas"]
    )

    tests = json.load(open("evaluation/test_cases.json", "r", encoding="utf-8"))

    passed = 0
    failed = 0

    for test in tests:
        print(f"Test {test['id']}: {test['description']}")
        print(f"Query: {test['query']}")

        result = pipeline.run(test["query"])

        # 1️⃣ Mode check
        if result.get("mode") != test["expected_mode"]:
            print(f"❌ Mode mismatch: got {result.get('mode')} | expected {test['expected_mode']}")
            failed += 1
            print("-" * 50)
            continue

        # 2️⃣ Decision check
        if "expected_decision" in test:
            if result.get("decision") != test["expected_decision"]:
                print(f"❌ Decision mismatch: got {result.get('decision')} | expected {test['expected_decision']}")
                failed += 1
                print("-" * 50)
                continue

        # 3️⃣ Answer content check
        if "expected_contains" in test:
            answer = result.get("answer", "").lower()
            if test["expected_contains"].lower() not in answer:
                print(f"❌ Answer mismatch: expected '{test['expected_contains']}' in answer")
                failed += 1
                print("-" * 50)
                continue

        print("✅ PASS")
        passed += 1
        print("-" * 50)

    print(f"\n=== Evaluation Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")


if __name__ == "__main__":
    run_evaluation()
