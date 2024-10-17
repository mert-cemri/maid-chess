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

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="TERMINATE",
)

as1 = autogen.AssistantAgent(
    name="Assistant 1",
    llm_config=llm_config,
)
as2 = autogen.AssistantAgent(
    name="Assistant 2",
    system_message="Explore what Assistant 1 did not",
    llm_config=llm_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, as1, as2], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="Find who won 2020 Rolland Garros Mens Singles."
)