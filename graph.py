# graph.py
import os
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from typing_extensions import TypedDict
from typing import List
import pdfplumber
import faiss
import gc # garbage collection

from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from memory_utils import monitor_memory
from agent import create_summary_chain, create_grade_chain, create_decomposition_chain, rerank_docs, hyde_expansion, summarize_image

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

class GraphState(TypedDict):
    docs : List[str] # list of chunked documents
    relevant_docs : List[str]
    previous_generation : str
    feedback : str 
    query : str
    generated_summary : str
    filepaths : List[str] # list of filepaths for the pdfs
    sparse_retriever: object
    dense_retriever: object
    counter: int  # number of times agent has attempted to generate a summary
    top_k: int # number of top documents to retrieve when ranking

def create_graph(callback_manager=None):
    workflow = StateGraph(GraphState)

    summary_chain = create_summary_chain(callback_manager)
    grade_chain = create_grade_chain(callback_manager)
    decomposition_chain = create_decomposition_chain(callback_manager)

    workflow.add_node("extract", extract)
    workflow.add_node("index", index)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("grade_summary", grade_summary)

    workflow.add_edge(START, 'extract')
    workflow.add_edge('extract', 'index')
    workflow.add_edge('index', 'retrieve')
    workflow.add_edge('retrieve', 'generate_summary')
    workflow.add_edge('generate_summary', 'grade_summary')
    workflow.add_conditional_edges('grade_summary', should_regenerate, {'generate_summary': 'generate_summary', 'end': END})

    graph = workflow.compile()        
    return graph

# document processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Nodes
def extract(state):
    monitor_memory("extract_start", state)

    print("---EXTRACTING PDF AND CREATING CHUNKS---")
    filepath_list = state['filepaths']
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    texts = []

    for i, filepath in enumerate(filepath_list):
        with pdfplumber.open(filepath) as pdf:
            text = ''
            for page_number, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    # If no text is extracted, use OCR
                    print(f"Page {page_number + 1} of {filepath} requires OCR")
                    images = convert_from_path(filepath, first_page=page_number + 1, last_page=page_number + 1)
                    page_content = []
                    for image in images:
                        ocr_text = summarize_image(image)
                        print(ocr_text)
                        page_content.append(ocr_text)
                    text += "\n".join(page_content)

            texts.append(text)
            monitor_memory(f"pdf_load_{i}", state)

    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type='gradient')
    docs = []
    for text in texts:
        docs.extend(text_splitter.create_documents([text]))
        monitor_memory(f"chunking_document_{i}", state)

    gc.collect()
    monitor_memory("extract_complete", state)
    
    return {**state, "docs": docs}


def index(state):
    monitor_memory("indexing_start", state)
    print("---INDEXING---")

    docs = state['docs']

    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    monitor_memory("pre_vectorstore", state)
    try:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        monitor_memory("post_vectorstore", state)

        dense_retriever = vectorstore.as_retriever(search_type="mmr") # maximum marginal relevance
        monitor_memory("post_dense_retriever", state)
        
        sparse_retriever = BM25Retriever.from_documents(docs, k=8)
        monitor_memory("post_sparse_retriever", state)

        gc.collect()
        monitor_memory("indexing_complete", state)
    
        return {**state, "dense_retriever": dense_retriever, "sparse_retriever": sparse_retriever}
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        # Monitor memory if error occurs
        monitor_memory("indexing_error", state)
        raise
    finally:
        # Cleanup original docs if they're no longer needed
        del docs
        gc.collect()


def retrieve(state):
    monitor_memory("retrieve_start", state)
    print("---RETRIEVING RELEVANT DOCUMENTS---")

    query = state['query']
    sparse_retriever = state['sparse_retriever']
    dense_retriever = state['dense_retriever']
    current_relevant_docs = state.get('relevant_docs', [])

    monitor_memory("pre_search_setup", state)

    search_kwargs = {
        "k": 12,  # Gradually increase k
        'fetch_k': 30,  # Gradually increase fetch_k
        'lambda_mult': 0.8  # Gradually decrease lambda_mult
    }

    dense_retriever.search_kwargs = search_kwargs
    hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.5, 0.5])

    monitor_memory("pre_query_processing", state)

    print("---DECOMPOSING QUERY AND IMPLEMENTING HYPOTHETICAL DOCUMENT EXPANSION---")

    try:
        query_decomposer = create_decomposition_chain()
        queries = query_decomposer.invoke({"query": query})
        queries = [query] + queries.queries
        
        gc.collect()
        monitor_memory("post_query_decomposition", state)

        for i, query in enumerate(queries):
            expanded_query = hyde_expansion(query)
            relevant_docs = hybrid_retriever.invoke(expanded_query)
            print(f"Found {len(relevant_docs)} relevant docs")

            relevant_docs = [doc for doc in relevant_docs if doc not in current_relevant_docs]
            current_relevant_docs += relevant_docs

            gc.collect()

            monitor_memory(f"query_batch_{i}", state)
    
        print(f"Obtained {len(current_relevant_docs)} relevant docs")
        monitor_memory("pre_reranking", state)

        reranked_docs = rerank_docs(current_relevant_docs, query, state['top_k'])

        print(f"Ranked {len(reranked_docs)} relevant docs")    

        gc.collect()
        monitor_memory("retrieve_complete", state)
        return {**state, "relevant_docs": reranked_docs, "counter":state.get('counter', 0), "previous_generation": "", "feedback": ""}

    except Exception as e:
       print(f"Error during retrieval: {str(e)}")
       monitor_memory("retrieve_error", state)
       raise
    finally:
       # Clean up large objects
       del current_relevant_docs
       gc.collect()


def generate_summary(state):
    monitor_memory("generate_summary_start", state)
    print(f"---GENERATION ATTEMPT {state['counter'] + 1}---")
    try:
        docs = state['relevant_docs']
        monitor_memory("pre_generation", state)
        context = format_docs(docs)
        del docs # clean up large objects
        gc.collect()
        summary_chain = create_summary_chain()
        generation = summary_chain.invoke(
            {
                "context": context,
                "previous_generation": state['previous_generation'],
                "feedback": state['feedback']
                }
                )
        monitor_memory("post_generation", state)
        gc.collect()

        return {**state, "generated_summary": generation}
    except Exception as e:
       print(f"Error during summary generation: {str(e)}")
       monitor_memory("generation_error", state)
       raise
    finally:
       # Clean up large objects
       del context 
       gc.collect()
       monitor_memory("generate_summary_complete", state)

def grade_summary(state):
    print("---GRADING---")
    monitor_memory("grade_summary_start", state)
    try:
        context = state['relevant_docs']
        generated_summary = state['generated_summary']
        monitor_memory("pre_grading", state)
        grade_chain = create_grade_chain()
        output = grade_chain.invoke(
        {
            "context": state['relevant_docs'],
            "generated_summary": state['generated_summary']
        }
    )
        monitor_memory("post_grading", state)
        gc.collect()
        return {
            **state,
            'previous_generation':state['generated_summary'],
            'binary_score': output.binary_score,
            'feedback': output.feedback,
            'counter': state['counter'] + 1
            }
    except Exception as e:
       print(f"Error during grading: {str(e)}")
       monitor_memory("grading_error", state)
       raise
    finally:
       # Clean up any large objects
       del context
       del generated_summary
       gc.collect()
       monitor_memory("grade_summary_complete", state)        

def should_regenerate(state):
    if state['counter'] > 2:
        return "end"
    if state['binary_score'] == 'no':
        return 'generate_summary'
    return "end"          
