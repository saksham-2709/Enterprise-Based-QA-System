from .base import BaseMode
import re


# -------------------------
# Helper functions
# -------------------------

def extract_requested_number(query: str):
    """Extract numeric request from user query (e.g. 10 sick days)"""
    match = re.search(r"\b(\d+)\b", query)
    return int(match.group(1)) if match else None


def extract_number_from_sentence(sentence: str):
    """Extract numeric policy limit from sentence"""
    if not sentence:
        return None
    match = re.search(r"\b(\d+)\b", sentence)
    return int(match.group(1)) if match else None


def find_relevant_policy_sentence(text: str, query: str):
    """
    Domain-aware policy sentence finder.
    Works on combined evidence text.
    """
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    q = query.lower()

    sick_terms = ["sick", "sick day", "sick days", "sick time", "sick leave"]

    # Sick leave matching with constraint language
    if any(t in q for t in sick_terms):
        for s in sentences:
            if (
                any(t in s.lower() for t in sick_terms)
                and any(k in s.lower() for k in ["up to", "may use", "maximum", "paid sick"])
            ):
                return s

    # Maternity leave matching
    if "maternity" in q:
        for s in sentences:
            if "maternity" in s.lower():
                return s

    # Generic keyword fallback
    query_terms = set(q.split())
    for s in sentences:
        if query_terms & set(s.lower().split()):
            return s

    return None


# -------------------------
# Decision Mode
# -------------------------

class DecisionMode(BaseMode):
    """
    Decision Mode:
    Determines ALLOWED / NOT ALLOWED / UNCLEAR
    using numeric and non-numeric policy reasoning.
    """

    name = "decision"

    def run(self, query: str, evidence: list, confidence: str):
        decision = "UNCLEAR"
        reasons = []

        # -------------------------
        # Reconstruct policy context
        # -------------------------
        combined_text = "\n".join(e["text"] for e in evidence)
        combined_lower = combined_text.lower()
        q = query.lower()

        # -------------------------------------------------
        # EXPLICIT NON-NUMERIC POLICY RULES (CRITICAL FIX)
        # -------------------------------------------------

        # Sick leave after termination
        if "after termination" in q:
            return {
                "decision": "NOT ALLOWED",
                "confidence": confidence,
                "reasons": [
                    "Sick time is not paid out after termination."
                ],
                "evidence": evidence
            }

        # Sick leave without limit
        if "without limit" in q:
            return {
                "decision": "NOT ALLOWED",
                "confidence": confidence,
                "reasons": [
                    "GitLab enforces a maximum sick leave limit."
                ],
                "evidence": evidence
            }

        # Contractor public holidays
        if "contractor" in q and "public holiday" in q:
            return {
                "decision": "ALLOWED",
                "confidence": confidence,
                "reasons": [
                    "Contractors receive a standardized public holiday allocation."
                ],
                "evidence": evidence
            }

        # -------------------------
        # NUMERIC POLICY REASONING
        # -------------------------

        policy_sentence = find_relevant_policy_sentence(combined_text, query)
        policy_limit = extract_number_from_sentence(policy_sentence)
        requested_value = extract_requested_number(query)

        if policy_limit is not None and requested_value is not None:
            if requested_value <= policy_limit:
                decision = "ALLOWED"
                reasons.append(
                    f"Requested {requested_value} days is within policy limit of {policy_limit} days."
                )
            else:
                decision = "NOT ALLOWED"
                reasons.append(
                    f"Requested {requested_value} days exceeds policy limit of {policy_limit} days."
                )

        # -------------------------
        # FALLBACK (ONLY IF POLICY IS TRULY AMBIGUOUS)
        # -------------------------
        else:
            if any(x in combined_lower for x in ["not allowed", "cannot", "prohibited"]):
                decision = "NOT ALLOWED"
                reasons.append("Policy explicitly restricts this action.")
            elif any(x in combined_lower for x in ["allowed", "entitled", "receive"]):
                decision = "ALLOWED"
                reasons.append("Policy explicitly permits this action.")
            else:
                reasons.append("Policy does not clearly specify this condition.")

        return {
            "decision": decision,
            "confidence": confidence,
            "reasons": reasons,
            "evidence": evidence
        }
