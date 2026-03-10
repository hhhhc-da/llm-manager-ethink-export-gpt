from fastapi import FastAPI, HTTPException, Depends, Header
import uvicorn
import time
from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message
from interface import ChatCompletionRequest, ChatCompletionResponse, ModelListResponse, Model
import os
import yaml

from agent import LargeLanguageModelManager

app = FastAPI()

BASE_CONFIG_PATH = "cfg"
YAML_PATH = os.path.join(BASE_CONFIG_PATH, "oumi.yaml")
CONFIG_PATH = os.path.join(BASE_CONFIG_PATH, "token.yaml")
LLM_CONFIG_PATH = os.path.join(BASE_CONFIG_PATH, "config.yaml")

API_KEY = ""
llm_manager: LargeLanguageModelManager = None

try:
    BASE_CONFIG = InferenceConfig.from_yaml(YAML_PATH)
    print(f"成功从 YAML 加载配置：{YAML_PATH}")
    print(f"加载的模型：{BASE_CONFIG.model.model_name}")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        TOKEN = yaml.safe_load(f.read())
        API_KEY = TOKEN['token']
    
    llm_manager = LargeLanguageModelManager(
        llm_model='zhipuai',
        yaml_path=LLM_CONFIG_PATH,
        oumi_yaml_path=YAML_PATH
    )
except Exception as e:
    print(f"加载 YAML 配置失败：{str(e)}")
    raise

def get_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="未提供 API Key")
    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer" or not param:
        raise HTTPException(status_code=401, detail="无效的 API Key 格式")
    if param != API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return param

def clean_input_text(text: str) -> str:
    while "USER: USER:" in text:
        text = text.replace("USER: USER:", "USER:")
    while "USER: , USER:" in text:
        text = text.replace("USER: , USER:", "USER:")
    while "USER: ，USER:" in text:
        text = text.replace("USER: ，USER:", "USER:")
    while "USER: ， USER:" in text:
        text = text.replace("USER: ， USER:", "USER:")
    return text.strip()

def extract_assistant_reply(result) -> str:
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
        assistant_reply = extract_assistant_reply(str(result))
    return assistant_reply

def conversation_to_text(conversation: Conversation) -> str:
    text_parts = []
    for msg in conversation.messages:
        role = msg.role.upper()
        content = clean_input_text(msg.content)
        text_parts.append(f"{role}: {content}")
    return "\n".join(text_parts) + "\nASSISTANT: "

@app.get("/v1/models", dependencies=[Depends(get_api_key)])
def list_models() -> ModelListResponse:
    models_data = llm_manager.get_supported_models()
    models = [Model(**m) for m in models_data]
    return ModelListResponse(data=models, object="list")

@app.post("/v1/chat/completions", dependencies=[Depends(get_api_key)])
def chat_completions_openai(request: ChatCompletionRequest) -> ChatCompletionResponse:
    yaml_model = BASE_CONFIG.model.model_name

    oumi_messages = []
    for msg in request.messages:
        role = str(msg.role).lower()
        content = clean_input_text(str(msg.content).strip())
        if not content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")
        oumi_messages.append(Message(role=role, content=content))
    conversation = Conversation(messages=oumi_messages)
    input_text = conversation_to_text(conversation)
    print(f"✓ 清理后的推理输入：\n{input_text}")

    try:
        user_question = ""
        for msg in reversed(request.messages):
            if msg.role.lower() == "user":
                user_question = msg.content
                break
        if not user_question:
            raise ValueError("未找到用户消息")
        
        model_switch_map = {
            "gpt-5": "zhipuai", # 我在 Dify 内配置的 gpt-5 接口, 修改这个即可
            "zhipuai-chatglm-4-flash": "zhipuai",
            BASE_CONFIG.model.model_name: "oumi",
            "deepseek-r1": "deepseek-r1"
        }

        # 如果请求的模型不在映射中，默认使用 oumi
        target_model = model_switch_map.get(request.model, "oumi")
        llm_manager.change_llm_model(target_model)
        
        if target_model == "zhipuai":
            assistant_reply = llm_manager.ask_function(user_question)
        elif target_model == "deepseek-r1":
            assistant_reply = llm_manager.ask_function(
                question=user_question,
                temperature=request.temperature if request.temperature else 0.4,
                max_tokens=request.max_tokens if request.max_tokens else 512
            )
        else:
            assistant_reply = llm_manager.ask_function(
                question=user_question,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        print(f"✓ 推理内容：{assistant_reply}")

    except Exception as e:
        error_detail = f"推理/过滤失败：{str(e)}"
        print(f"✗ {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    return ChatCompletionResponse(
        model=request.model,
        created=int(time.time()),
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_reply
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(input_text),
            "completion_tokens": len(assistant_reply),
            "total_tokens": len(input_text) + len(assistant_reply)
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=83, reload=False)