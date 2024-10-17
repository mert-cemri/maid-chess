import random  # noqa E402

import matplotlib.pyplot as plt  # noqa E402
import networkx as nx  # noqa E402

import autogen  # noqa E402
from autogen.agentchat.conversable_agent import ConversableAgent  # noqa E402
from autogen.agentchat.assistant_agent import AssistantAgent  # noqa E402
from autogen.agentchat.groupchat import GroupChat  # noqa E402
from autogen.graph_utils import visualize_speaker_transitions_dict  # noqa E402

# llama3.1:70b
config_list = [
  {
    "model": "llama3.1:70b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
  }
]
config_list_llama = {
    "timeout": 600,
    "cache_seed": 0,  # change the seed for different trials
    "config_list": config_list,
    "temperature": 1,
}

# agents = [ConversableAgent(name=f"Agent{i}", llm_config=False) for i in range(3)]
orchestrator = ConversableAgent(
        name="Orchestrator",
        system_message='''You are an orchestrator agent managing 2 assistants to solve a chess problem.
        You need to provide the optimal chess move given a position.
        You need to ask the problem to your assistants, get their answers, and make a final verdict on their answers.
        Do not attempt to solve the problem yourself first, listen to your assistants.
        At every round, aggregate your assistants findings, and deduct what you think the optimal move is. Make sure that is a valid move.
        At every output, clearly state what the optimal move is with your reasoning and ask them again if they want to change their opinion.
        Once you are sure of an answer, then terminate the discussion using TERMINATE. State the optimal move before stating TERMINATE''',
        llm_config=config_list_llama,
        )
assistant1 = ConversableAgent(
        name="Assistant 1",
        system_message='''You are a chess expert. 
        Your job is to provide the optimal move given a chess board position. 
        Return your answer in the chess notation, i.e. when knight on f6 captures the pawn on d5, only and only return the string 'Nxd5'.
        
        Here is the position overview:
        White: Zc4 Qb7 d6 Z6
        Black: Kf8 b6.

        Z means there is a piece in that location, but you dont know what the piece is.
        
        Once you are sure of an answer and colloborated with agent 2, then terminate the discussion using TERMINATE.
        '''
        ,
        llm_config=config_list_llama,
    )
assistant2 = ConversableAgent(
        name="Assistant 2",
        system_message='''You are a chess expert. 
        Your job is to provide the optimal move given a chess board position. 
        Return your answer in the chess notation, i.e. when knight on f6 captures the pawn on d5, only and only return the string 'Nxd5'.

        Here is the position overview:
        White: Zc4 Qb7 Z6 e6
        Black: Zf8 b6.

        Z means there is a piece in that location, but you dont know what the piece is.
        ''',
        llm_config=config_list_llama,
        # max_consecutive_auto_reply=3
    )
# agents = [orchestrator, assistant1, assistant2]
# allowed_speaker_transitions_dict = {
#     agents[0]: [agents[1], agents[2]],
#     agents[1]: [agents[0]],
#     agents[2]: [agents[0]],
# }
agents = [assistant1, assistant2]
allowed_speaker_transitions_dict = {
    agents[0]: [agents[0],agents[1]],
    agents[1]: [agents[0],agents[1]],
}
plt.figure(figsize=(6, 3))
visualize_speaker_transitions_dict(allowed_speaker_transitions_dict, agents)

chess_prompt = """
Below is a description of a chess board position. White to move. Find the optimal move for the whites.
Return your answer in the precise chess notation, i.e. when knight on f6 captures the pawn on d5, only and only return the string 'Nxd5'.

Position Overview (White to move):
White: Kf1 Qf4 Rc1 Bb3 Ne3 a4 d4 f2 g2 h2
Black: Kg8 Qc7 Ra8 Re8 Bf6 Nd7 a5 b6 d5 e6 g7 h7
"""
# White: Ke1 Qc2 Rc1 Rh1 Bd2 Bd3 Nf3 a2 e3 e5 f2 g2 h2. 
# Black: Kh8 Qb6 Ra8 Rf8 Bb7 Be7 Nh6 a6 b5 e6 f7 g7.

chess_prompt_2 = """
Below is a description of a chess board position. White to move. Find the optimal move for the white. There is a move that ends in check-mate. Find that.
Return your answer in the precise chess notation, i.e. when knight on f6 captures the pawn on d5, only and only return the string 'Nxd5'.

Position Overview (White to move):
White: Kc4 Qb7 e6 d6
Black: Kf8 b6.
"""

chess_prompt_3 = """
Below is a description of a chess board position. White to move. Find the optimal move for the white. There is a move that ends in check-mate. Find that.
Return your answer in the precise chess notation, i.e. when knight on f6 captures the pawn on d5, only and only return the string 'Nxd5'.
"""
# White: Kc4 Qb7 e6 d6
# Black: Kf8 b6

def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=30,
    allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict,
    speaker_transitions_type="allowed",
)
manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=config_list_llama,
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
)

agents[0].initiate_chat(
    manager,
    message=chess_prompt_3,
)