import os
from dotenv import load_dotenv
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import pymupdf as fitz
from PIL import Image
import io
import pandas as pd

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio

# Force event loop in non-main threads
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 4096
MODEL_NAME = "gemini-1.5-flash-latest"
TEMPERATURE = 0.4

# Get Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_output_tokens=MAX_TOKENS
)

# QA Prompt template
PROMPT = PromptTemplate(
    template="""Context: {context}

Question: {question}

Answer the question concisely based on the given context. 
If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

If the question is about images or tables, refer to them specifically in your answer.""",
    input_variables=["context", "question"]
)

def process_pdf(pdf_file):
    """Extract text from PDF and split into chunks."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    
    print(f"Total extracted text length: {len(text)}")

    # ✅ Explicit separator for better splitting
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    texts = text_splitter.split_text(text)
    print(f"Number of text chunks created: {len(texts)}")  # Debugging
    return texts

def extract_images_and_tables(pdf_file):
    """Extract images and tables from PDF."""
    doc = fitz.open(pdf_file)
    images = []
    tables = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((f"Page {page_num + 1}, Image {img_index + 1}", image))

        # Extract tables if any
        tables_on_page = page.find_tables()
        for table_index, table in enumerate(tables_on_page):
            df = pd.DataFrame(table.extract())
            tables.append((f"Page {page_num + 1}, Table {table_index + 1}", df))

    return images, tables



from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio

def create_embeddings_and_vectorstore(texts):
    async def _make_embeddings():
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        return FAISS.from_texts(texts, embeddings)

    return asyncio.run(_make_embeddings())



def expand_query(query: str, llm_model) -> str:
    """Expand the original query with related terms."""
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""Given the following query, generate 3-5 related terms or phrases 
        that could be relevant to the query. Separate the terms with commas.

        Query: {query}

        Related terms:"""
    )
    chain = LLMChain(llm=llm_model, prompt=prompt)
    response = chain.run(query)
    expanded_terms = [term.strip() for term in response.split(',')]
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    return expanded_query

def rag_pipeline(query, qa_chain, vectorstore, images, tables):
    """Run the RAG (Retrieval-Augmented Generation) pipeline."""
    expanded_query = expand_query(query, llm)
    relevant_docs = vectorstore.similarity_search_with_score(expanded_query, k=3)

    context = ""
    log = "Query Expansion:\n"
    log += f"Original query: {query}\n"
    log += f"Expanded query: {expanded_query}\n\n"
    log += "Relevant chunks:\n"
    for i, (doc, score) in enumerate(relevant_docs, 1):
        context += doc.page_content + "\n\n"
        log += f"Chunk {i} (Score: {score:.4f}):\n"
        log += f"Sample: {doc.page_content[:200]}...\n\n"

    # Add info about images and tables
    context += f"Number of images in the PDF: {len(images)}\n"
    context += f"Number of tables in the PDF: {len(tables)}\n"

    response = qa_chain.invoke({"query": query})
    return response['result'], log

def process_pdf_and_query(pdf_file, query):
    """Process PDF and handle the query."""
    texts = process_pdf(pdf_file)
    images, tables = extract_images_and_tables(pdf_file)
    vectorstore = create_embeddings_and_vectorstore(texts)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    result, chunks_log = rag_pipeline(query, qa, vectorstore, images, tables)

    return result, len(texts), len(images), len(tables), chunks_log

def gradio_interface(pdf_file, query):
    """Gradio interface function."""
    result, num_chunks, num_images, num_tables, chunks_log = process_pdf_and_query(pdf_file.name, query)

    log = f"PDF processed successfully.\n"
    log += f"Number of text chunks: {num_chunks}\n"
    log += f"Number of images extracted: {num_images}\n"
    log += f"Number of tables extracted: {num_tables}\n\n"
    log += chunks_log

    return result, log

def main():
    """Main function to launch the Gradio interface."""
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.File(label="Upload PDF"),
            gr.Textbox(label="Enter your question")
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Processing Log")
        ],
        title="PDF Question Answering System (Gemini)",
        description="Upload a PDF and ask questions about its content, including images and tables. Powered by Google Gemini."
    )

    iface.launch()

if __name__ == "__main__":
    main()
