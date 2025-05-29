from typing import List, Dict, Any
from pydantic import BaseModel
from judgeval.common.tracer import Tracer, wrap
from judgeval import JudgmentClient
from judgeval.scorers import ToolOrderScorer, ToolDependencyScorer
from judgeval.common.tracer import Tracer
import os


judgment = Tracer(project_name="multi_agent_system")
judgment_client = JudgmentClient()

class Message(BaseModel):
    sender: str
    content: str
    recipient: str

@judgment.identify(identifier="name")
class SimpleAgent:
    def __init__(self, name: str, tracer: Tracer):
        self.name = name
        self.tracer = tracer
        self.messages: List[Message] = []
        
    @judgment.observe(span_type="tool")
    def send_message(self, content: str, recipient: str) -> None:
        """
        Send a message to another agent

        Args:
            content: The content of the message to send.
            recipient: The name of the recipient of the message.
        """
        message = Message(sender=self.name, content=content, recipient=recipient)
        self.messages.append(message)
        return f"Message sent to {recipient}: {content}"
    
    @judgment.observe(span_type="function")
    def receive_message(self, sender: str) -> List[str]:
        """Get all messages from a specific sender"""
        received = [msg.content for msg in self.messages if msg.sender == sender]
        return received
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages this agent has received"""
        return self.messages

class MultiAgentSystem:
    def __init__(self):
        self.tracer = Tracer()
        self.agents: Dict[str, SimpleAgent] = {}
    
    def add_agent(self, name: str) -> SimpleAgent:
        """Add a new agent to the system"""
        agent = SimpleAgent(name, self.tracer)
        self.agents[name] = agent
        return agent
    
    @judgment.observe(span_type="function")
    def run_simple_task(self, prompt: str):
        """Run a simple task where agents communicate with each other"""
        # Create two agents
        alice = self.add_agent("Alice")
        bob = self.add_agent("Bob")
        
        # Have them exchange messages

        alice.send_message("Hello Bob, how are you?", "Bob")
        bob.send_message("I'm good Alice, thanks for asking!", "Alice")
        # Print the conversation
        print("\nAlice's messages:")
        for msg in alice.get_all_messages():
            print(f"From {msg.sender}: {msg.content}")
            
        print("\nBob's messages:")
        for msg in bob.get_all_messages():
            print(f"From {msg.sender}: {msg.content}")

# Example usage
if __name__ == "__main__":
    system = MultiAgentSystem()
    system.run_simple_task("Do something random")

    # test_file = os.path.join(os.path.dirname(__file__), "tests.yaml")
    # judgment_client.assert_test(
    #     scorers=[ToolOrderScorer()],
    #     function=system.run_simple_task,
    #     tracer=judgment,
    #     override=True,
    #     test_file=test_file,
    #     eval_run_name="multi_agent_tool_order",
    #     project_name="multi_agent_system"
    # )

    tools = [
        {
            "type": "function",
            "name": "send_message",
            "description": "Send a message to another agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "self": {
                        "type": "SimpleAgent",
                        "description": "The name of the agent sending the message",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the message to send.",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "The name of the recipient of the message.",
                    },
                },
                "required": ["self", "content", "recipient"],
            }
        }
    ]

    test_file2 = os.path.join(os.path.dirname(__file__), "tests2.yaml")
    judgment_client.assert_test(
        scorers=[ToolDependencyScorer(enable_param_checking=True)],
        function=system.run_simple_task,
        tracer=judgment,
        override=True,
        test_file=test_file2,
        eval_run_name="multi_agent_tool_dependency",
        project_name="multi_agent_system"
    )
