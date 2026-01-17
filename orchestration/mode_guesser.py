class ModeGuesser:
    def guess(self, query: str) -> str:
        q = query.lower().strip()

        # 1️⃣ Search intent
        if any(k in q for k in [
            "policy", "document", "rules", "guidelines",
            "search", "find", "show"
        ]):
            return "search"

        # 2️⃣ Decision intent = permission + restriction
        decision_constraints = [
            "allowed",
            "not allowed",
            "permitted",
            "eligible",
            "exceed",
            "beyond",
            "can i",
            "can we",  
            "restriction",
            "constraint",
            "prohibited",
            "take more than",
            "can employee",
            "without limit",
            "after termination",
            "post termination","public holiday"
        ]

        # Case A: explicit restriction language
        if "can a" in q and any(c in q for c in decision_constraints):
            return "decision"

        # Case B: numeric enforcement (e.g. 10 sick days)
        if "can a" in q and any(char.isdigit() for char in q):
            return "decision"

        # 3️⃣ QA intent (default)
        return "qa"
