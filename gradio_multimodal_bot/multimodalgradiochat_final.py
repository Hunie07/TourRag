import os
import io
import time
import base64
import pickle
import gradio as gr
from PIL import Image
from base64 import b64decode
from langchain_chroma import Chroma
from IPython.display import HTML, display
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# ChatOpenAI 모델 초기화
chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def is_base64(s):
    """문자열이 Base64로 인코딩되었는지 확인합니다"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False
    
def image_to_base64(image_path):
    """
    주어진 이미지 파일 경로를 읽어 base64 형식의 문자열로 변환합니다.
    
    Args:
        image_path (str): 이미지 파일의 경로
        
    Returns:
        str: base64 인코딩된 문자열
    """
    with open(image_path, "rb") as image_file:
        # 이미지 파일을 바이너리로 읽어 base64로 인코딩합니다.
        encoded_bytes = base64.b64encode(image_file.read())
    # bytes를 문자열로 변환하여 반환합니다.
    base64_string = encoded_bytes.decode("utf-8")
    return base64_string

def plt_img_base64(image_base64):
    """
    Base64로 인코딩된 이미지를 주피터 노트북에 표시
    
    매개변수:
    img_base64 (str): Base64로 인코딩된 이미지 문자열
    """
    # Base64 문자열을 소스로 하는 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{image_base64}" />'

    # HTML을 렌더링하여 이미지 표시
    display(HTML(image_html))

def summarize_image(base64_string):
    prompt_template = """
    Please analyze the following image in detail and provide a thorough description.
    The image is related to a tourist attraction in Korea, and if it contains any Korean text, please transcribe it exactly as it appears.
    Your description should cover aspects such as location, architectural style, surrounding scenery, cultural elements, and the overall atmosphere conveyed by the image.
    Do not provide any explanations other than the output format.
    Do not say that you are not unable to analyze the image. If you cannot analyze the image, provide a description of the image itself.
    **Use 한국어.**

    [Output Format]
    1. 주요 건축물: 
    2. 주변 환경: 
    3. 문화적 요소: 
    4. 한국어 텍스트: 
    5. 기타: 
    """

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    image_chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

    # 이미지 요약 생성
    image_summary = image_chain.invoke(base64_string)
    return image_summary

# InMemoryStore 로드하기
def load_store_from_disk(path: str):
    # 파일에서 데이터를 로드
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # 새 InMemoryStore 생성
    new_store = InMemoryStore()
    
    # 로드된 데이터로 초기화
    new_store.store = data  # 직접 store 속성에 할당
    
    return new_store

# Store 로드
# loaded_store = load_store_from_disk("mm_summaries.pkl")
loaded_store = load_store_from_disk("mm_summaries_gemma3_12b.pkl")
id_key = "doc_id"

# 벡터 저장소 로드
vectorstore = Chroma(
    collection_name="mm_summaries", 
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="../chroma_db",  # 벡터 저장소 경로
)

# 문서 저장소 초기화
docstore = InMemoryStore()

# 로드한 저장소로 새 검색기 만들기
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  
    docstore=loaded_store,
    id_key=id_key,
    search_kwargs={"k": 5},
)

# 대화 이력 관리를 위한 설정
max_items = 16
conversation_history = []

# 시스템 메시지 설정
system_message = SystemMessage(
    content="한국어로 대화하는 AI 어시스턴트입니다. 이미지와 텍스트를 함께 처리할 수 있으며, 관련 문서도 검색하여 답변합니다."
    )
conversation_history.append(system_message)

def process_prompt(kwargs):
    """문맥과 질문을 기반으로 프롬프트를 구성합니다"""
    # 검색된 문서(텍스트와 이미지)와 사용자 질문 추출
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    
    # 텍스트 문맥 구성
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element

    # 텍스트와 이미지를 포함한 프롬프트 템플릿 구성
    prompt_template = f"""
    Based solely on the provided context, answer the question. The context may include text and image summary.
    If an image summary is provided as input, please refer to the image summary and respond with similar content.
    
    [Context]
    {context_text}

    [Question]
    {user_question}

    [Answer (in 한국어)]
    """
    # 프롬프트 콘텐츠 초기화 (텍스트로 시작)
    prompt_content = [{"type": "text", "text": prompt_template}]

    # 최종 ChatPromptTemplate 생성 및 반환
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def base64_to_pil(b64_string):
    # 만약 데이터 URL 형식이라면 헤더 부분 제거
    if b64_string.startswith("data:image"):
        header, b64_string = b64_string.split(",", 1)
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))

def split_image_text_types(docs):
    """검색결과 내 이미지와 텍스트를 분리합니다"""
    images = []
    texts = []
    for doc in docs:
        texts.append(doc['title'] + ': \n- 분류: ' + doc['cat2'] + '\n\n- 개요: ' + doc['overview'])
        if is_base64(doc['image']):
            try:
                # base64 문자열을 PIL 이미지로 변환
                pil_image = base64_to_pil(doc['image'])
                images.append(pil_image)
            except Exception as e:
                images.append(None)
        else:
            images.append(None)
    return {"texts": texts, "images": images}

def search_documents(query):
    """벡터 스토어에서 문서를 검색하는 함수"""
    search_pipeline = retriever | RunnableLambda(split_image_text_types)
    search_results = search_pipeline.invoke(query)
    return search_results

def print_like_dislike(x: gr.LikeData):
    """좋아요/싫어요 처리 함수"""
    print(x.index, x.value, x.liked)

def add_message(history, message):
    """사용자 메시지를 처리하고 대화 이력에 추가하는 함수"""
    # 두 입력(파일과 텍스트)이 모두 있는 경우
    if message.get("files") and message.get("text"):
        history.append({"role": "user", "content": {"path": message["files"][0]}})
        # 파일 처리: 첫 번째 파일 경로 사용
        file_path = message["files"][0]
        base64_string = image_to_base64(file_path)
        image_summary = summarize_image(base64_string)
        
        # 텍스트 입력
        text_input = message["text"]
        # 이미지 요약과 텍스트 입력을 결합하여 검색 쿼리 구성
        combined_query = f"{text_input}\n{image_summary}"
        
        relevant_docs = search_documents(combined_query)
        context = "\n".join(relevant_docs["texts"])
        
        # UI에 표시할 메시지 구성 (필요에 따라 변경)
        ui_message = text_input + " [이미지 첨부됨]"
        history.append({"role": "user", "content": ui_message})
        conversation_history.append(HumanMessage(content=f"Context: {context}\n\nUser question: {ui_message}"))
        
        images = relevant_docs["images"]
        texts = relevant_docs["texts"]
    
    # 파일만 있는 경우
    elif message.get("files"):
        history.append({"role": "user", "content": {"path": message["files"][0]}})
        file_path = message["files"][0]
        base64_string = image_to_base64(file_path)
        image_summary = summarize_image(base64_string)

        relevant_docs = search_documents(image_summary)
        context = "\n".join(relevant_docs["texts"])
        
        ui_message = message.get("text", "이미지 업로드")
        history.append({"role": "user", "content": ui_message})
        conversation_history.append(HumanMessage(content=f"Context: {context}\n\nUser question: {ui_message}"))
        
        images = relevant_docs["images"]
        texts = relevant_docs["texts"]
    
    # 텍스트만 있는 경우
    elif message.get("text"):
        text_input = message["text"]
        relevant_docs = search_documents(text_input)
        context = "\n".join(relevant_docs["texts"])
        
        ui_message = text_input
        history.append({"role": "user", "content": ui_message})
        conversation_history.append(HumanMessage(content=f"Context: {context}\n\nUser question: {ui_message}"))
        
        images = relevant_docs["images"]
        texts = relevant_docs["texts"]
    
    # 5개의 검색 결과를 각 컴포넌트에 매핑 (만약 검색 결과가 부족하면 None 혹은 빈 문자열)
    output_images = [images[i] if i < len(images) else None for i in range(5)]
    output_texts = [texts[i] if i < len(texts) else "" for i in range(5)]

    # 기존에는 gr.MultimodalTextbox를 리셋하는 식으로 반환했으므로 그대로 사용하고,
    # 오른쪽 컬럼에 해당하는 5개의 이미지와 텍스트 컴포넌트를 순서대로 반환합니다.
    return history, gr.MultimodalTextbox(value=None, interactive=True), \
           output_images[0], output_texts[0], \
           output_images[1], output_texts[1], \
           output_images[2], output_texts[2], \
           output_images[3], output_texts[3], \
           output_images[4], output_texts[4]

def reset_chat():
    global conversation_history
    # 대화 내역 초기화 및 시스템 메시지 재추가
    conversation_history.clear()
    conversation_history.append(system_message)
    # 챗봇 대화 내역과 입력창을 초기화합니다.
    return [], gr.MultimodalTextbox(value=None, interactive=True)

def bot(history: list):
    """AI 어시스턴트의 응답을 생성하는 함수"""
    # API 호출하여 응답 생성
    response = chat.invoke([msg for msg in conversation_history])
    content = response.content
    
    # 응답을 대화 이력에 추가
    history.append({"role": "assistant", "content": ""})
    conversation_history.append(AIMessage(content=content))
    
    # 스트리밍 효과를 위한 점진적 출력
    for character in content:
        history[-1]["content"] += character
        time.sleep(0.01)
        yield history
    
    # 대화 이력 크기 제한
    if len(conversation_history) > max_items:
        conversation_history.pop(1)  # 시스템 메시지는 유지

# Gradio 인터페이스 구성
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="red")) as app:
    with gr.Row():               
        with gr.Column():
            chatbot = gr.Chatbot(
                label="멀티모달 챗봇",
                elem_id="chatbot",
                bubble_full_width=False,
                height=750,
                type="messages",
            )

            # 멀티모달 입력 구성
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="single",
                placeholder="메시지를 입력하거나 파일을 업로드하세요...",
                show_label=False,
                #sources=["upload"],
            )
            
            reset_btn = gr.Button("초기화")

        with gr.Column():
            # 오른쪽 컬럼: 이미지와 텍스트 컴포넌트를 미리 정의
            result_image_1 = gr.Image(label='검색된 이미지')
            result_text_1  = gr.Textbox(label='관광지 설명')
            result_image_2 = gr.Image(label='검색된 이미지')
            result_text_2  = gr.Textbox(label='관광지 설명')
            result_image_3 = gr.Image(label='검색된 이미지')
            result_text_3  = gr.Textbox(label='관광지 설명')
            result_image_4 = gr.Image(label='검색된 이미지')
            result_text_4  = gr.Textbox(label='관광지 설명')
            result_image_5 = gr.Image(label='검색된 이미지')
            result_text_5  = gr.Textbox(label='관광지 설명')

            # 좌측의 채팅 메시지 제출 이벤트 설정
            # 추후 봇 응답 체인과 연결합니다.
            # 여기서는 이후에 오른쪽 컬럼 업데이트를 위해 추가 출력들을 함께 리턴합니다.
            # 즉, add_message 의 리턴값을 history, chat_input, 그리고 5개의 이미지, 5개의 텍스트로 구성합니다.
    chat_msg = chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input,  # 기존 두 출력
            # 오른쪽 컬럼 업데이트: 5개의 이미지와 5개의 텍스트 컴포넌트를 출력에 포함
            # 아래 변수들은 우측 컬럼에서 생성한 컴포넌스 리스트 (아래쪽 참고)
            result_image_1, result_text_1,
            result_image_2, result_text_2,
            result_image_3, result_text_3,
            result_image_4, result_text_4,
            result_image_5, result_text_5]
    )
    
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    reset_btn.click(reset_chat, outputs=[chatbot, chat_input])

# 앱 실행
if __name__ == "__main__":
    app.queue()
    app.launch(debug=True)