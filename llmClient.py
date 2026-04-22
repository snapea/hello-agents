import os 
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class HelloAgentsLLM:
    """
    定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or os.getenv("LLM_TIMEOUT", 60)

        print(f"使用模型: {self.model}")
        print(f"API Key: {'已设置' if apiKey else '未设置'}")
        print(f"Base URL: {baseUrl}")

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """

        print(f"正在调用{self.model}模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            print("✅模型响应成功")
            collected_response = []
            for chunk in response: 
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", "")
                if not delta:
                    continue
                content = getattr(delta, "content", "")
                if not content:
                    continue           
                print(content, end="", flush=True)  # 实时输出流式响应
                collected_response.append(content)
            print()  # 换行
            return "".join(collected_response)
        except Exception as e:
            print(f"调用模型时发生错误: {e}")
            raise

if __name__ == "__main__":
    print('项目初始化')
    try:
        llmClient = HelloAgentsLLM()
        exampleMessage = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code"},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessage)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)
    
    except ValueError as e:
        print(f"初始化LLM客户端失败: {e}")