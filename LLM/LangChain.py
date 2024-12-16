from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import textwrap
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, "data.txt")

llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.8, "max_length":512})

text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

with open(data_file_path, "r", encoding="utf-8") as file:
    data = file.read()

documents = text_splitter.split_text(data)

docs = [Document(page_content=doc) for doc in documents[:4]]

chain = load_summarize_chain(llm, chain_type="stuff")

prompt_template = prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""


prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

summary = chain.run(docs)

wrapped_text = textwrap.fill(summary, width=110)

print(docs[0].page_content)

print("Respuesta:")
print(wrapped_text)


##CATEGORIZACION

prompt_template = prompt_template = """Describe the content according to this perspectives
social
economica
politica
cultural
educativa
salud
medio ambiente
transporte
infraestructura
{text}
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

summary = chain.run(docs)

wrapped_text = textwrap.fill(summary, width=110)

print("Categoria:")

print(wrapped_text)




