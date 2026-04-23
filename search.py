import os 
from serpapi import SerpApiClient
from dotenv import load_dotenv

load_dotenv()

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try: 
        apiKey = os.getenv("SERPAPI_API_KEY")
        if not apiKey:
            return "错误:LLM_API_KEY 未在 .env 文件中配置。"
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": apiKey,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        client = SerpApiClient(params)
        ret = client.get_dict()
        
        
        # 优先返回直接答案
        if "answer_box_list" in ret:
            return "\n".join(ret["answer_box_list"])
        if "answer_box" in ret and "answer" in ret["answer_box"]:
            return ret["answer_box"]["answer"]
        if "knowledge_graph" in ret and "description" in ret["knowledge_graph"]:
            return ret["knowledge_graph"]["description"]
        if "organic_results" in ret and ret["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(ret["organic_results"][:3])
            ]
            return "\n".join(snippets)
            
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
    

if __name__ == "__main__":
    search('Python编程语言的最新发展')