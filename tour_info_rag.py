from dotenv import load_dotenv
load_dotenv()

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

import io
import base64
from io import BytesIO
from PIL import Image
from IPython.display import HTML, display
from langchain_core.documents import Document

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
    return encoded_bytes.decode("utf-8")

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

import pickle
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.stores import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# InMemoryStore 저장하기
def save_store_to_disk(store: InMemoryStore, path: str):
    # InMemoryStore에는 _data가 아닌 store 속성이 있습니다
    data = dict(store.store)  # store.store에 접근
    
    # 데이터를 직렬화하여 파일에 저장
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Store saved to {path}")

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
loaded_store = load_store_from_disk("mm_summaries.pkl")
# loaded_store = load_store_from_disk("C:\Users\hoony\SeSAC_Winter\project\mm_summaries.pkl")
id_key = "doc_id"

# 벡터 저장소 로드
vectorstore = Chroma(
    collection_name="mm_summaries", 
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",  # 벡터 저장소 경로
)

# 로드한 저장소로 새 검색기 만들기
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  
    docstore=loaded_store,
    id_key=id_key,
    search_kwargs={"k": 5},
)

from base64 import b64decode
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def process_prompt(kwargs):
    """문맥과 질문을 기반으로 프롬프트를 구성합니다"""
    # 검색된 문서(텍스트와 이미지)와 사용자 질문 추출
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    print(f"검색된 문서 개수: {len(docs_by_type['texts'])}")
    print(f"검색된 이미지 개수: {len(docs_by_type['images'])}")
    print("-" * 100)
    

    # 텍스트 문맥 구성
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element

    # 텍스트와 이미지를 포함한 프롬프트 템플릿 구성
    prompt_template = f"""
    Based solely on the provided context, answer the question. The context may include text and images below.
    If an image summary is provided as input, please refer to the image summary and respond with similar content.

    [Context]
    {context_text}

    [Question]
    {user_question}

    [Answer (in 한국어)]
    """

    # 프롬프트 콘텐츠 초기화 (텍스트로 시작)
    prompt_content = [{"type": "text", "text": prompt_template}]

    # 이미지가 있으면 프롬프트에 추가
    # if len(docs_by_type["images"]) > 0:
    #     for image in docs_by_type["images"]:
    #         prompt_content.append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    #             }
    #         )

    # 최종 ChatPromptTemplate 생성 및 반환
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def split_image_text_types(docs):
    """검색결과 내 이미지와 텍스트를 분리합니다"""
    images = []
    texts = []
    for doc in docs:
        texts.append(doc['title'] + ': \n- 분류: ' + doc['cat2'] + '\n- 개요: ' + doc['overview'] + '\n\n' + '- 이미지 요약:\n' + doc['image_summary'])
        if is_base64(doc['image']):
            # 서버 오류를 방지하기 위해 이미지 크기 조정 
            images.append(doc['image'])  # base64로 인코딩된 문자열
        else:
            pass
    return {"texts": texts, "images": images}

# 소스를 포함한 확장 RAG 파이프라인
rag_chain_with_sources = {
    "context": retriever | RunnableLambda(split_image_text_types),  # 검색기로 문서 가져와서 타입별로 분류
    "question": RunnablePassthrough(),  # 사용자 질문 그대로 전달
} | RunnablePassthrough().assign(  # 원본 입력을 유지하면서 응답 필드 추가
    response=(
        RunnableLambda(process_prompt)  # 분류된 문서와 질문으로 프롬프트 구성
        | ChatOpenAI(model="gpt-4o-mini")  # LLM으로 응답 생성
        | StrOutputParser()  # 응답을 문자열로 변환
    )
)

from langfuse.callback import CallbackHandler

# 콜백 핸들러 생성
langfuse_handler = CallbackHandler()

# RAG 파이프라인 실행
result = rag_chain_with_sources.invoke("출렁다리 있을까?", config={"callbacks": [langfuse_handler]})

print(result["response"])
print("-" * 100)
for image in result['context']['images']:
    print(image)
    plt_img_base64(image)
    print("=" * 100)