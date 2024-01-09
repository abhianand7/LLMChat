import torch
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# model4bit = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config, use_safetensors=True)
model4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)



generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.max_new_tokens = 256
generation_config.temperature=0.6
generation_config.top_p = 0.95
generation_config.repetition_penalty = 1.0
generation_config.do_sample = True

pipeline = pipeline(
    "text-generation",
    model=model4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
#     max_length=1500,
    num_return_sequences=1,
    generation_config=generation_config,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id= tokenizer.eos_token_id
)

# lets format prompt so as to get desired results and put in place some guardrails
# def format_prompt(message, history):
#     system_prompt = "you are an AI assistant for Radisys Company who reads texts and answers questions about them and who always assists with care, respect and truth. Respond with utmost utility yet securely. Avoid, harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
#     prompt = "<s>"
    
#     for user_prompt, bot_response in history:
#         prompt += f"[INST] {user_prompt} [/INST]"
#         prompt += f" {bot_response}</s> "
#     prompt += f"[INST] {message} [/INST]"
#   return prompt

llm = HuggingFacePipeline(pipeline=pipeline)


# from langchain import PromptTemplate, LLMChain

# template = """[INST] You are a helpful, respectful and honest assistant. Answer exactly from the context
# Answer the question below from the context below:
# {context}
# {question} [/INST]
# """
# question_p = """What is the date for the announcement"""
# context_p = """ On August 10, it was announced that its subsidiary, JSW Neo Energy, has agreed to acquire a portfolio encompassing 1753 megawatts of renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""

# prompt = PromptTemplate(template=template, input_variables=["question", "context"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# response = llm_chain.run({"question": question_p, "context": context_p})


# lets create a prompt template and use that inside retrieval chain
template = """
[INST]<>
you are an AI assistant for Radisys Company who reads context and answers questions about them in a short and crisp manner and who always assists with care, respect and truth. Respond with utmost utility yet securely. 
Avoid, harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
<>

{context}

{question} [/INST]
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

pdf_loader = PyPDFLoader('Rsys_India_EmployeeHandbook_01_Apr_2019_Ver3.1 1.pdf')
pages = pdf_loader.load()

print(len(pages))

print(pages[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
splits = text_splitter.split_documents(pages)

# st_model_id = "sentence-transformers/all-mpnet-base-v2"

st_model_id = "sentence-transformers/gtr-t5-large"

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embedding_model = HuggingFaceEmbeddings(
    model_name=st_model_id, 
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


chroma_db = Chroma.from_documents(documents=splits, embedding=embedding_model)

search_kwargs={"k":2}
retriever = chroma_db.as_retriever(search_kwargs=search_kwargs)

retriever_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": prompt}
)


"""
1. what is the paternity leave policy for men at Radisys?
"""

text_query = "tell me about referral bonus"
response = retriever_qa.invoke(text_query)
print(response)


def llm_agent(prompt:str, history) -> str:
    yield retriever_qa.invoke(prompt)


# def authenticate(username, password):
#     print(username)
#     print(password)
#     if username == r"admin" and password == r"Pwd@admin@07":
#         return True

# iface = gr.ChatInterface(
#     llm_agent,
#     chatbot=gr.Chatbot(height=500),
#     textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
#     title="Radisys HR Tool",
#     description="Ask me any question on Radisys Employee Policy",
#     theme="soft",
# #     examples=["What is the leave policy?", "What is the dress code?",
# #     cache_examples=True,
#     retry_btn=None,
#     undo_btn="Delete Previous",
#     clear_btn="Clear",
# ).queue()
# iface.launch(share=True, auth=authenticate)

