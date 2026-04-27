from llmClient import HelloAgentsLLM
from reActAgent import ReActAgent
from tool import ToolExecutor
from searchUtil import search  # 你已有的工具

if __name__ == "__main__":
    print("🚀 启动 ReAct Agent")

    # 1. 初始化 LLM
    llm = HelloAgentsLLM()

    # 2. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 3. 注册工具
    toolExecutor.registerTool(
        name="Search",
        description="用于搜索信息的工具",
        func=search
    )

    # 4. 初始化 Agent
    agent = ReActAgent(llm=llm, tools=toolExecutor)

    # 5. 提问
    question = "小米最新手机"

    result = agent.run(question)

    print("\n\n✅ 最终结果：")
    print(result)