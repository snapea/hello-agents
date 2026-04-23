from tool import ToolExecutor
from llmClient import HelloAgentsLLM
from search import re

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm: HelloAgentsLLM, tools: ToolExecutor, maxSteps: int = 5):
        self.llm = llm
        self.tools = tools
        self.maxSteps = maxSteps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        currentStep = 0
        while currentStep < self.maxSteps:
            currentStep += 1
            print(f"--- 第 {currentStep} 步 ---")

            # 1. 格式化提示词
            toolsDesc = self.tools.getAvailableTools() # 获取工具描述
            historyStr = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools = toolsDesc,
                question = question,
                history = historyStr
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

    
    def parseOutput(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """

        # Thought: 匹配到 Action: 或文本末尾
        thoughtMatch = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)

        # Action: 匹配到文本末尾
        actionMatch = re.search(r"Action:\s*(.*)", text, re.DOTALL)

        thought = thoughtMatch.group(1).strip() if thoughtMatch else None
        action = actionMatch.group(1).strip() if actionMatch else None
        return thought, action

    def parseAction(self, actionText: str):
        """解析Action字符串，提取工具名称和输入。
        """

        match = re.match(r"(\w+)\[(.*)\]", actionText, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
