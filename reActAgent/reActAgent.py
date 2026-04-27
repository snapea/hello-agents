from tool import ToolExecutor
from llmClient import HelloAgentsLLM
from searchUtil import search
import re

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
    def __init__(self, llm: HelloAgentsLLM, tools: ToolExecutor, maxSteps: int = 10):
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
        print(f"🚀 当前最大步数: {self.maxSteps}")
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
            responseText = self.llm.think(messages=messages)

            if not responseText:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self.parseOutput(responseText)

            if thought:
                print(f"思考: {thought}")
            
            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    finalAnswer = match.group(1)
                    print(f"🎉 最终答案: {finalAnswer}")
                    return finalAnswer
                else:
                    print(f"警告:无法解析Finish指令: {action}")
                    break
            
            toolName, toolInput = self.parseAction(action)
            if not toolName or not toolInput:
                print(f"警告:无法解析Action '{action}'，流程终止。")
                continue

            print(f"🎬 行动: {toolName}[{toolInput}]")
            
            toolFunc = self.tools.getTool(toolName)
            if not toolFunc:
                observation = f"错误:未找到名为 '{toolName}' 的工具。"
            else:
                observation = toolFunc(toolInput)

            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    
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
