from cerebrum.agents.base import BaseAgent
from cerebrum.llm.communication import LLMQuery
import json
import traceback
import logging
import datetime

class MathAgent(BaseAgent):
    def __init__(self, agent_name, task_input, config_):
        try:
            super().__init__(agent_name, task_input, config_)

            self.plan_max_fail_times = 3
            self.tool_call_max_fail_times = 3

            self.start_time = None
            self.end_time = None
            self.request_waiting_times: list = []
            self.request_turnaround_times: list = []
            self.task_input = task_input
            self.messages = []
            self.workflow_mode = "manual"  # (manual, automatic)
            self.rounds = 0
            self.status = "initialized"  # 添加状态跟踪
            self.debug_logs = []  # 用于收集调试信息
            self._log_debug(f"MathAgent initialized with name: {agent_name}, task: {task_input}")
            self._log_debug(f"Initial status: {self.status}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize MathAgent: {str(e)}")
            print(traceback.format_exc())
            raise

    def _log_debug(self, message: str):
        """记录调试信息到实例变量"""
        log_entry = {
            "type": "debug",
            "message": message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print(f"[DEBUG] {message}")  # 立即打印到控制台
        self.debug_logs.append(log_entry)

    def _log_error(self, message: str, error: Exception = None):
        """记录错误信息到实例变量"""
        error_info = {
            "type": "error",
            "message": message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if error:
            error_info["error"] = str(error)
            error_info["traceback"] = traceback.format_exc()
        print(f"[ERROR] {message}")  # 立即打印到控制台
        if error:
            print(f"[ERROR] Exception: {str(error)}")
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        self.debug_logs.append(error_info)

    def _update_status(self, new_status: str):
        """Helper method to update and log status changes"""
        old_status = self.status
        self.status = new_status
        message = f"Status changed: {old_status} -> {new_status}"
        self._log_debug(message)

    def get_status(self):
        """返回当前agent的状态信息"""
        try:
            status_info = {
                "agent_name": self.agent_name,
                "status": self.status,
                "rounds": self.rounds,
                "workflow_mode": self.workflow_mode,
                "debug_logs": self.debug_logs,  # 包含调试日志
                "timestamp": datetime.datetime.now().isoformat()
            }
            self._log_debug("Status requested")
            print(f"[DEBUG] Full status info: {json.dumps(status_info, indent=2)}")
            return status_info
        except Exception as e:
            error_msg = f"Error getting status: {str(e)}"
            self._log_error(error_msg, e)
            error_info = {
                "agent_name": self.agent_name,
                "status": "error",
                "error": str(e),
                "debug_logs": self.debug_logs,
                "timestamp": datetime.datetime.now().isoformat()
            }
            return error_info

    def run(self):
        try:
            self._update_status("running")
            self._log_debug(f"Starting run with task: {self.task_input}")
            
            if not self.build_system_instruction():
                error_result = {
                    "agent_name": self.agent_name,
                    "result": "Failed to build system instruction",
                    "rounds": self.rounds,
                    "status": self.status,
                    "debug_logs": self.debug_logs
                }
                return error_result

            task_input = self.task_input
            self.messages.append({"role": "user", "content": task_input})

            workflow = None

            if self.workflow_mode == "automatic":
                workflow = self.automatic_workflow()
                self.messages = self.messages[:1]  # clear long context
            else:
                workflow = self.manual_workflow()

            if not workflow:
                self._update_status("failed")
                error_result = {
                    "agent_name": self.agent_name,
                    "result": "Failed to generate a valid workflow.",
                    "rounds": self.rounds,
                    "status": self.status,
                    "debug_logs": self.debug_logs
                }
                return error_result

            self.messages.append(
                {
                    "role": "user",
                    "content": f"[Thinking]: The workflow generated for the problem is {json.dumps(workflow)}. Follow the workflow to solve the problem step by step. ",
                }
            )

            final_result = ""
            self._update_status("executing_workflow")

            for i, step in enumerate(workflow):
                action_type = step["action_type"]
                action = step["action"]
                tool_use = step["tool_use"]

                self._log_debug(f"Executing step {i+1}: {action}")
                prompt = f"At step {i + 1}, you need to: {action}. "
                self.messages.append({"role": "user", "content": prompt})

                if tool_use:
                    selected_tools = self.pre_select_tools(tool_use)
                else:
                    selected_tools = None

                response = self.send_request(
                    agent_name=self.agent_name,
                    query=LLMQuery(
                        messages=self.messages,
                        tools=selected_tools,
                        action_type=action_type,
                    ),
                )["response"]
                
                self.messages.append({"role": "assistant", "content": response.response_message})
                self.rounds += 1

            final_result = self.messages[-1]["content"]
            self._update_status("completed")
            
            success_result = {
                "agent_name": self.agent_name,
                "result": final_result,
                "rounds": self.rounds,
                "status": self.status,
                "debug_logs": self.debug_logs
            }
            return success_result

        except Exception as e:
            self._update_status("error")
            self._log_error("Error during execution", e)
            
            error_result = {
                "agent_name": self.agent_name,
                "result": f"Error occurred during execution: {str(e)}",
                "rounds": self.rounds,
                "status": self.status,
                "error": str(e),
                "debug_logs": self.debug_logs
            }
            return error_result

    def build_system_instruction(self):
        try:
            self._update_status("building_system_instruction")
            prefix = "".join(["".join(self.config["description"])])

            plan_instruction = "".join(
                [
                    f"You are given the available tools from the tool list: {json.dumps(self.tool_info)} to help you solve problems. ",
                    "Generate a plan with comprehensive yet minimal steps to fulfill the task. ",
                    "The plan must follow the json format as below: ",
                    "[",
                    '{"action_type": "action_type_value", "action": "action_value","tool_use": [tool_name1, tool_name2,...]}',
                    '{"action_type": "action_type_value", "action": "action_value", "tool_use": [tool_name1, tool_name2,...]}',
                    "...",
                    "]",
                    "In each step of the planned plan, identify tools to use and recognize no tool is necessary. ",
                    "Followings are some plan examples. ",
                    "[" "[",
                    '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
                    '{"action_type": "chat", "action": "write a summarization based on the gathered information. ", "tool_use": []}',
                    "];",
                    "[",
                    '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
                    '{"action_type": "chat", "action": "understand the current methods and propose ideas that can improve ", "tool_use": []}',
                    "]",
                    "]",
                ]
            )

            if self.workflow_mode == "manual":
                self.messages.append({"role": "system", "content": prefix})
            else:
                assert self.workflow_mode == "automatic"
                self.messages.append({"role": "system", "content": prefix})
                self.messages.append({"role": "user", "content": plan_instruction})
            
            self._update_status("system_instruction_built")
            return True
        except Exception as e:
            self._log_error("Error in build_system_instruction", e)
            self._update_status("error")
            return False

    def manual_workflow(self):
        try:
            self._update_status("generating_workflow")
            workflow = [
                {
                    "action_type": "tool_use",
                    "action": "Search for relevant mathematical concepts and formulas",
                    "tool_use": ["demo_author/arxiv"],
                },
                {
                    "action_type": "chat",
                    "action": "Analyze the mathematical problem and provide solution steps",
                    "tool_use": [],
                },
                {
                    "action_type": "chat",
                    "action": "Calculate and verify the final answer",
                    "tool_use": [],
                }
            ]
            self._update_status("workflow_generated")
            return workflow
        except Exception as e:
            self._log_error("Error in manual_workflow", e)
            self._update_status("error")
            return None

    def automatic_workflow(self):
        try:
            self._update_status("generating_workflow")
            for i in range(self.plan_max_fail_times):
                response = self.send_request(
                    agent_name=self.agent_name,
                    query=LLMQuery(
                        messages=self.messages, tools=None, message_return_type="json"
                    ),
                )["response"]

                workflow = self.check_workflow(response.response_message)

                self.rounds += 1

                if workflow:
                    self._update_status("workflow_generated")
                    return workflow
                else:
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan",
                        }
                    )
            self._update_status("workflow_generation_failed")
            return None
        except Exception as e:
            self._log_error("Error in automatic_workflow", e)
            self._update_status("error")
            return None

    def __str__(self):
        """String representation of the agent's current state"""
        return f"MathAgent(name={self.agent_name}, status={self.status}, rounds={self.rounds})"

