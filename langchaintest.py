from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
# 导入文本
# loader = UnstructuredFileLoader("/content/sample_data/data/lg_test.txt")
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')
llm = Ollama(model="llama2")
# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
# llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# # 创建总结链
# chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# # 执行总结链，（为了快速演示，只总结前5段）
# chain.run(split_documents[:5])
embeddings = OllamaEmbeddings() 
docsearch = Chroma.from_documents(split_documents, embeddings)
# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
# 进行问答
vector = FAISS.from_documents(split_documents, embeddings)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
 
<context>
{context}
</context>
 
Question: {input}""")
 
document_chain = create_stuff_documents_chain(llm, prompt)
# result = qa({"query": "科大讯飞今年第三季度收入是多少？"})
from langchain.chains import create_retrieval_chain
 
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# docs = loader.load()
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
 
# LangSmith提供了几个功能，可以帮助进行测试:...

# model_id = 'google/flan-t5-large'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# pipe = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=100
# )

# local_llm = HuggingFacePipeline(pipeline=pipe)
# print(local_llm('What is the capital of France? '))


# template = """Question: {question} Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm = local_llm
# llm_chain = prompt | llm
# question = "What is the capital of England?"
# print(llm_chain.invoke({"question": question}))
