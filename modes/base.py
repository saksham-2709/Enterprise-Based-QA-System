class BaseMode:
    name = None
    top_k = 5
    confidence_threshold = 0.4
    use_llm = True

    def run(self, query, retriever):
        raise NotImplementedError
