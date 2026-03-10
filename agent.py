# coding = utf-8
import os
from typing import Literal, Optional, List, Dict, Any
from zhipuai import ZhipuAI
import re
import yaml
from copy import copy
from llama_cpp import Llama
from oumi.core.configs import InferenceConfig, GenerationParams
from oumi.core.types.conversation import Conversation, Message
from oumi.infer import infer
import time

"""
统一接口管理大语言模型的交互

--------------------------------------
                Models
--------------------------------------
- ZhipuAI 的 ChatGLM-4-Flash 模型 (通过 API 访问)
- 本地部署的 DeepSeek-R1 模型 (通过 llama-cpp 接口访问)
- 本地部署的 SmolLM2-135M-Instruct 模型 (通过 oumi 推理框架访问)
--------------------------------------

模型统一使用 Llama-2 的聊天格式, 以保证输入输出的一致性和兼容性
请避免占用太多核心数导致其他服务运行出现异常
"""

class LargeLanguageModelManager():
    """
    大语言模型接口类, 延迟初始化模型实例以节省内存
    切换模型时立即初始化新模型，切换出时销毁旧模型
    """
    def __init__(self, 
                 llm_model:Literal['deepseek-r1', 'zhipuai', 'oumi']='zhipuai', 
                 yaml_path:str=os.path.join('cfg', 'config.yaml'),
                 oumi_yaml_path: str = os.path.join('cfg', 'oumi.yaml'),
                 deepseek_model_path: str = os.path.join("models", "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
                 ):
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 配置文件未找到: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.llm_model = llm_model
        self.oumi_yaml_path = oumi_yaml_path
        self.deepseek_model_path = deepseek_model_path

        self.deepseek_serve = DeepSeekServe()
        self.oumi_serve = OumiServe()
        self.chatglm_model = None

        self.prompt = self._load_prompt_template()

        self.ask_function = None
        # 初始化时立即创建指定模型实例
        self._init_ask_function(init_model=True)

    def _load_prompt_template(self) -> List[Dict[str, str]]:
        """
        加载 Prompt 模板
        """
        prompt = []
        prompt_path = self.config['llm'].get('prompt', "")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                while line := f.readline().strip():
                    if line.startswith("(System)"):
                        prompt.append({"role": "system", "content": line[len("(System)"):].strip()})
                    elif line.startswith("(User)"):
                        prompt.append({"role": "user", "content": line[len("(User)"):].strip()})
                    elif line.startswith("(Assistant)"):
                        prompt.append({"role": "assistant", "content": line[len("(Assistant)"):].strip()})
                    else:
                        print(f"无法识别的行格式: {line}")
        print(f"成功加载 Prompt 模板: \n{prompt}\n")
        return prompt

    def _init_ask_function(self, init_model: bool = True):
        """
        初始化对应模型的调用函数
        :param init_model: 是否立即初始化模型实例
        """
        if self.llm_model == 'zhipuai':
            if init_model:
                self._init_zhipuai()
            self.ask_function = self.chatglm_response
        elif self.llm_model == 'deepseek-r1':
            if init_model:
                self.deepseek_serve.initialize(
                    chat_format='llama-2', 
                    llama_path=self.deepseek_model_path
                )
            self.ask_function = self.deepseek_response
        elif self.llm_model == 'oumi':
            if init_model:
                self.oumi_serve.initialize(yaml_path=self.oumi_yaml_path)
            self.ask_function = self.oumi_response

    def _init_zhipuai(self):
        """
        初始化 ZhipuAI（仅在切换到该模型时）
        """
        self.api_key = self.config['llm'].get('chatglm-api', "")
        if self.api_key:
            self.chatglm_model = ZhipuAI(api_key=self.api_key)
            print("成功初始化 ZhipuAI ChatGLM-4-Flash 模型")
        else:
            raise RuntimeError("zhipuai 模型未配置 API Key，无法使用")

    def change_llm_model(self, new_model:Literal['deepseek-r1', 'zhipuai', 'oumi']):
        """
        切换语言模型, 清理旧模型资源, 立即初始化新模型
        """
        if new_model == self.llm_model:
            print(f"当前已经是 {new_model} 模型, 无需切换")
            return
        
        # 清理旧模型资源
        if self.llm_model == 'deepseek-r1':
            self.deepseek_serve.cleanup()
        elif self.llm_model == 'oumi':
            self.oumi_serve.cleanup()
        elif self.llm_model == 'zhipuai':
            self.chatglm_model = None
            print("已销毁 ZhipuAI 模型实例")
        
        self.llm_model = new_model
        self._init_ask_function(init_model=True)

    def get_supported_models(self) -> List[Dict[str, Any]]:
        """
        获取支持的模型列表（供 FastAPI 接口使用）
        """
        # 先初始化 Oumi 配置以获取模型名称（只读配置，不加载模型）
        if not self.oumi_serve._initialized:
            try:
                temp_config = InferenceConfig.from_yaml(self.oumi_yaml_path)
                oumi_model_name = temp_config.model.model_name
            except:
                oumi_model_name = "oumi-unknown"
        else:
            oumi_model_name = self.oumi_serve.base_config.model.model_name

        models = [
            {
                "id": "zhipuai-chatglm-4-flash",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ZhipuAI"
            },
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "DeepSeek"
            },
            {
                "id": oumi_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Oumi"
            }
        ]
        return models

    def deepseek_response(self, question: str, temperature:float=0.4, max_tokens:int=512) -> str:
        """
        DeepSeek 模型响应
        """
        print(f"开始询问 DeepSeek: {question}")
        
        message = copy(self.prompt)
        message.append({"role": "user", "content": question})
        
        response = self.deepseek_serve.generate(
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = response.strip()
        print(f"DeepSeek 回复信息: {response}\n")
        return response

    def oumi_response(self, question: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        Oumi 模型响应（单条消息处理）
        """
        print(f"开始询问 Oumi: {question}")
        
        message = copy(self.prompt)
        message.append({"role": "user", "content": question})
        
        response = self.oumi_serve.infer(
            messages=message,
            max_tokens=max_tokens,
            temperature=temperature
        )
        response = response.strip()
        print(f"Oumi 回复信息: {response}\n")
        return response

    def chatglm_response(self, question: str) -> str:
        """
        ChatGLM 模型响应（单条消息处理）
        """
        print(f"开始询问 ChatGLM: {question}")
        
        message = copy(self.prompt)
        message.append({"role": "user", "content": question})
        
        response = self.chatglm_model.chat.completions.create(
            model="glm-4-flash",
            messages=message,
        ).choices[0].message.content
        
        response = response.strip()
        print(f"ChatGLM 回复信息: {response}\n")
        return response
    
class OumiServe:
    """
    Oumi 模型服务类，封装 Oumi 推理逻辑
    """
    def __init__(self):
        self.base_config: Optional[InferenceConfig] = None
        self.yaml_path: Optional[str] = None
        self._initialized = False

    def initialize(self, yaml_path: str = os.path.join("cfg", "oumi.yaml")):
        """
        延迟初始化 Oumi 配置
        """
        if self._initialized:
            return
        
        self.yaml_path = os.path.abspath(yaml_path)
        try:
            # 加载 Oumi 推理配置
            self.base_config = InferenceConfig.from_yaml(self.yaml_path)
            print(f"成功初始化 Oumi 模型：{self.base_config.model.model_name}")
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"初始化 Oumi 配置失败：{str(e)}")

    def clean_input_text(self, text: str) -> str:
        """
        清理输入文本中的重复 USER 标识
        """
        while "USER: USER:" in text:
            text = text.replace("USER: USER:", "USER:")
        while "USER: , USER:" in text:
            text = text.replace("USER: , USER:", "USER:")
        while "USER: ，USER:" in text:
            text = text.replace("USER: ，USER:", "USER:")
        while "USER: ， USER:" in text:
            text = text.replace("USER: ， USER:", "USER:")
        return text.strip()

    def extract_assistant_reply(self, result) -> str:
        """
        从 Oumi 推理结果中提取助手回复
        """
        assistant_reply = ""
        if isinstance(result, Conversation) and hasattr(result, "messages"):
            for msg in reversed(result.messages):
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    role = msg.role.lower()
                    content = msg.content.strip()
                    if role == "assistant" and content:
                        content = content.replace("ASSISTANT:", "").strip()
                        assistant_reply = content
                        break
        elif isinstance(result, str):
            if "ASSISTANT:" in result:
                parts = result.split("ASSISTANT:")
                assistant_reply = parts[-1].strip()
                filter_keywords = ["conversation_id", "metadata", "messages"]
                for keyword in filter_keywords:
                    if keyword in assistant_reply:
                        assistant_reply = assistant_reply.split(keyword)[0].strip()
        else:
            assistant_reply = self.extract_assistant_reply(str(result))
        return assistant_reply

    def conversation_to_text(self, conversation: Conversation) -> str:
        """
        将 Conversation 转换为 Oumi 所需的文本格式
        """
        text_parts = []
        for msg in conversation.messages:
            role = msg.role.upper()
            content = self.clean_input_text(msg.content)
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts) + "\nASSISTANT: "

    def infer(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        调用 Oumi 模型进行推理（核心方法）
        :param messages: 消息列表，格式 [{"role": "user/assistant/system", "content": "..."}]
        :param max_tokens: 最大生成token数
        :param temperature: 生成温度
        :return: 助手回复内容
        """
        if not self._initialized:
            raise RuntimeError("Oumi 模型未初始化，请先调用 initialize 方法")

        oumi_messages = []
        for msg in messages:
            role = str(msg["role"]).lower()
            content = self.clean_input_text(str(msg["content"]).strip())
            if not content:
                raise ValueError("消息内容不能为空")
            oumi_messages.append(Message(role=role, content=content))
        
        conversation = Conversation(messages=oumi_messages)
        input_text = self.conversation_to_text(conversation)
        print(f"✓ Oumi 清理后的推理输入：\n{input_text}")

        updated_generation = GenerationParams(
            max_new_tokens=max_tokens if max_tokens is not None else self.base_config.generation.max_new_tokens,
            batch_size=self.base_config.generation.batch_size,
            temperature=temperature if temperature is not None else self.base_config.generation.temperature,
            top_p=getattr(self.base_config.generation, "top_p", 1.0),
        )

        inference_config = InferenceConfig(
            model=self.base_config.model,
            generation=updated_generation,
            engine=self.base_config.engine
        )

        try:
            infer_results = infer(config=inference_config, inputs=[input_text])
            raw_result = infer_results[0] if infer_results else None
            assistant_reply = self.extract_assistant_reply(raw_result)
            return assistant_reply
        except Exception as e:
            raise RuntimeError(f"Oumi 推理失败：{str(e)}")

    def cleanup(self):
        """
        清理 Oumi 资源
        """
        self.base_config = None
        self.yaml_path = None
        self._initialized = False

class DeepSeekServe():
    """
    DeepSeek 后端接口类 (非流式输出版本)
    兼容 ZhipuAI/OpenAI 格式的 messages
    """
    def __init__(self):
        self.llama = None
        self.chat_format = None  
        self.llama_path = None
        self._initialized = False

    def initialize(
        self, 
        chat_format:Literal['llama-2']='llama-2',
        llama_path=os.path.join("models", "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    ):
        """
        延迟初始化 DeepSeek 模型（切换模型时才创建实例）
        """
        if self._initialized:
            return
        
        self.chat_format = chat_format
        self.llama_path = os.path.abspath(llama_path)

        if not os.path.exists(self.llama_path):
            raise FileNotFoundError(f"Llama 模型文件不存在: {self.llama_path}")
        
        self.llama = Llama(
            model_path=self.llama_path,
            n_ctx=512,
            n_threads=2,
            chat_format=self.chat_format,
            verbose=False
        )
        print(f"成功初始化 DeepSeek 模型 (chat 格式: {self.chat_format})")
        self._initialized = True

    def generate(self, messages, temperature:float=0.4, max_tokens:int=512, stop_tokens:list=["<tool_call>"]) -> str:
        """
        DeepSeek 生成接口（非流式）
        """
        if not self._initialized:
            raise RuntimeError("DeepSeek 模型未初始化, 请先调用 initialize")

        try:
            # 调用原生chat接口, stream=False 非流式输出
            result = self.llama.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens,
                stream=False
            )
            text = result["choices"][0]["message"]["content"].strip()
            return text
                
        except Exception as e:
            print(f"(Deepseek Generate) Exception: {e}")
            raise

    def cleanup(self):
        """
        清理模型资源, 释放内存
        """
        if self._initialized and self.llama is not None:
            del self.llama
            self.llama = None

        self.chat_format = None
        self.llama_path = None
        self._initialized = False

if __name__ == "__main__":
    llm_manager = LargeLanguageModelManager(llm_model='zhipuai')
    
    resp = llm_manager.ask_function("你好，请介绍一下自己，限制在 20 字以内。")
    print(f"ChatGLM 响应: {resp}\n")
    
    llm_manager.change_llm_model('deepseek-r1')
    resp = llm_manager.ask_function("你好，请介绍一下自己，限制在 20 字以内。")
    print(f"DeepSeek 响应: {resp}\n")
    
    llm_manager.change_llm_model('oumi')
    resp = llm_manager.ask_function("what is your model? Response in 10 words")
    print(f"Oumi 响应: {resp}\n")