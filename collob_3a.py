import autogen

from autogen import AssistantAgent, UserProxyAgent

config_list = [
  {
    "model": "llama3",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
  }
]
llm_config = {"config_list": config_list, "cache_seed": 42}

class ManagerAgent(UserProxyAgent):
    def __init__(self, config, assistants):
        super().__init__(config)
        self.assistants = assistants  # List of assistant agents

    def handle_user_input(self, user_input):
        # Manager receives the task from user, splits it into subtasks for assistants
        subtasks = self.create_subtasks(user_input)
        results = []
        for i, task in enumerate(subtasks):
            # Send subtask to corresponding assistant
            result = self.assistants[i].handle_task(task)
            results.append(result)
        # Compile results and send it back to user
        return self.compile_results(results)

    def create_subtasks(self, user_input):
        # Here, divide the user's input into smaller subtasks for the assistants
        # Example splitting based on a simple rule (you can modify this to fit your task)
        return [f"Subtask 1: {user_input}", f"Subtask 2: {user_input}"]

    def compile_results(self, results):
        # Compile and format the results from assistants before returning to the user
        return f"Results from assistants: {results[0]}, {results[1]}"

class AssistantAgent1(AssistantAgent):
    def __init__(self, config):
        super().__init__(config)

    def handle_task(self, task):
        # Use llama3 model to process the task
        response = self.complete(task)
        return response
    
assistant1 = AssistantAgent1(config_list[0])
assistant2 = AssistantAgent1(config_list[0])
manager = ManagerAgent(config_list[0], [assistant1, assistant2])

user_prompt = "Solve a complex problem and divide tasks"
response = manager.handle_user_input(user_prompt)
print(response)