from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, "data.txt")

loader = TextLoader(data_file_path, encoding="utf-8")
document = loader.load()

#Pre procesado

import textwrap

def wrap_text(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

embedding = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embedding)



#print(wrap_text(str(docs[0].page_content)))

llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.8, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

query = "Haz un resumen en español del texto"

docSummary = db.similarity_search(query)

#chain.run(input_documents=docSummary, question=query)
output = chain.invoke(input={"question": query, "input_documents": docSummary})

print("Respuesta")
print(output)

