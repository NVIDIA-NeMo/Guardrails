import asyncio

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from tests.v2_x.chat import ChatInterface

YAML_CONFIG = """
colang_version: "2.x"

models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
"""


async def run_chat_interface_based_on_script(script, colang, wait_time) -> str:
    rails_config = RailsConfig.from_content(
        colang_content=colang,
        yaml_content=YAML_CONFIG,
    )
    interaction_log = []
    rails_app = LLMRails(rails_config, verbose=True)
    chat = ChatInterface(rails_app)

    lines = script.split("\n")
    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            interaction_log.append(line)
            user_input = line.replace("> ", "")
            print(f"sending '{user_input}' to process")
            response = await chat.process(user_input, wait_time)
            interaction_log.append(response)

    chat.should_terminate = True
    await asyncio.sleep(0.5)

    return "\n".join(interaction_log)


def cleanup(content):
    output = []
    lines = content.split("\n")
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if line.strip() == ">":
            continue
        if line.startswith("#"):
            continue
        if "Starting the chat" in line:
            continue

        output.append(line.strip())

    return "\n".join(output)


async def compare_interaction_with_script(test_script, colang, wait_time=1.0):
    result = await run_chat_interface_based_on_script(test_script, colang, wait_time)
    clean_test_script = cleanup(test_script)
    clean_result = cleanup(result)
    assert (
        clean_test_script == clean_result
    ), f"\n----\n{clean_result}\n----\n\ndoes not match test script\n\n----\n{clean_test_script}\n----"
