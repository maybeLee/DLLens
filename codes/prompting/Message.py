class Message(object):
    def __init__(self, query: str = None, context: list = None):
        if context is None:
            context = []
        self.query = query
        self.context = context

    def update_system_message(self, system_message: str) -> None:
        self.context.append({'role': 'system', 'content': f'{system_message}'})

    def update_context(self, context: dict[str, str]) -> None:
        self.context = context

    def update_query(self, query):
        self.query = query

    def update_context_by_query_answer(self, query, answer):
        self.context.append({"role": "user", "content": f"{query}"})
        self.context.append({"role": "assistant", "content": f"{answer}"})

    def show_message(self):
        for item in self.message:
            if item["role"] == "system":
                print(f"System: \n{item['content']}")
            elif item["role"] == "user":
                print(f"User: \n{item['content']}")
            elif item["role"] == "assistant":
                print(f"Assistant: \n{item['content']}")
            
    @property
    def message(self):
        return self.form_message(self.query, self.context)

    @staticmethod
    def form_message(query: str, context: list = None) -> list:
        message = []
        if context is not None:
            for item in context:
                message.append(item)
        message.append(
            {"role": "user", "content": f"{query}"}
        )
        return message
