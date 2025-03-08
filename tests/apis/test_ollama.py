import unittest
from cerebrum.llm.apis import *


# The TestAgent class can test API llm_chat.
class TestAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []

    def run(self, task_input):
        self.messages.append({"role": "user", "content": task_input})

        tool_response = llm_chat(
            agent_name=self.agent_name,
            messages=self.messages,
            base_url="http://localhost:8000"
        )

        final_result = tool_response["response"]["response_message"]
        return final_result
    
    
    
    
   




class TestLLMAPI(unittest.TestCase):

    def setUp(self):
        self.agent = TestAgent("test_agent")


    def test_agent_with_greeting(self):
        response = self.agent.run("Hello, how are you?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agent_with_math_question(self):
        response = self.agent.run("What is 25 times 4?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agent_with_science_question(self):
        response = self.agent.run("Explain the theory of relativity.")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agent_with_history_question(self):
        response = self.agent.run("Who was the first president of the United States?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agent_with_technology_question(self):
        response = self.agent.run("What is quantum computing?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    

    
   
    
    
            
    


if __name__ == "__main__":
    unittest.main()
    

        
        

