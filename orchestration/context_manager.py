class ContextManager:
    def __init__(self):
        self.topic = None
        self.entity = None

    def update(self, query: str):
        q = query.lower()

        # Detect topic (simple + expandable)
        if "sick" in q:
            self.topic = "sick leave"
        elif "maternity" in q:
            self.topic = "maternity leave"
        elif "leave" in q:
            self.topic = "leave policy"

        # Detect entity references
        if "leave" in q:
            self.entity = self.topic

    def enrich_query(self, query: str) -> str:
        """
        Rewrite follow-up queries using context
        """
        q = query.lower()

        # Handle pronouns / vague references
        if any(x in q for x in ["it", "they", "those", "ones"]):
            if self.entity:
                return f"{query} related to {self.entity} policy"

        return query
