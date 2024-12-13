# agent.py
import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from functools import lru_cache
from PIL import Image
import io
import base64
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import anthropic

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

class GradeSummary(BaseModel):
    binary_score: str = Field(..., description="Summary meets all conditions relevant to a clinician, 'yes' or 'no'")
    feedback: str = Field(..., description="Explanation of why the summary does not meet all conditions")

class QueryDecomposer(BaseModel):
    queries: List[str] = Field(
        description="List of specific, searchable queries that together cover all aspects of the original query."
    )

@lru_cache(maxsize=1)
def get_reranker():
    """Cached initialization of the CrossEncoder to prevent reloading."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

class VisualProcessor:
    """Handles processing of images and PDFs"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.llm = ChatAnthropic(
            model='claude-3-5-sonnet-20241022',
            anthropic_api_key=api_key,
            temperature=0,
            top_p=0.7
        )
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> tuple:
        """
        Encode image to base64 and determine media type.
        Accepts file paths, PIL Images, or bytes.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            
        if isinstance(image, Image.Image):
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
                
            # Convert to bytes using JPEG format
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Use explicit media type for JPEG
            media_type = 'image/jpeg'
            
            # Encode to base64
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            return img_base64, media_type
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


    def process_single_image(self, image: Union[str, Path, Image.Image], prompt: str) -> str:
        """Process a single image """
        img_base64, media_type = self.encode_image(image)
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }]
        )
        
        return message.content[0].text

    def process_multiple_images(self, images: List[Union[str, Path, Image.Image]], prompt: str) -> str:
        """Process multiple images with """
        content = []
        
        for image in images:
            img_base64, media_type = self.encode_image(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_base64,
                }
            })
            
        # Add the prompt text
        content.append({
            "type": "text",
            "text": prompt
        })
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        return message.content[0].text

    def create_visual_chain(self, system_prompt: str = None):
        """Create a LangChain chain for image processing"""
        def process_input(inputs):
            if isinstance(inputs.get('image'), list):
                return self.process_multiple_images(
                    inputs['image'], 
                    inputs.get('prompt', "Analyze these medical images. Don't leave out any details.")
                )
            else:
                return self.process_single_image(
                    inputs['image'],
                    inputs.get('prompt', "Analyze this medical image. Don't leave out any details.")
                )
        
        from langchain.chains import LLMChain
        from langchain_core.runnables import RunnableLambda
        
        return RunnableLambda(process_input)

def create_medical_visual_processor(api_key=os.getenv('ANTHROPIC_API_KEY')):
    """Factory function to create a medical-focused visual processor"""
    processor = VisualProcessor(api_key)
    
    # Create chain with medical-specific prompt
    medical_prompt = """
    Analyze this medical document image with focus on:
    1. Patient information
    2. Diagnoses
    3. Test results
    4. Medications
    5. Treatment plans
    
    Be precise and comprehensive in extracting medical information. Do not leave out any details.
    """
    
    chain = processor.create_visual_chain(system_prompt=medical_prompt)
    return processor, chain
    

def summarize_image(image):
    """Create a summary of the image content."""
    _, visual_chain = create_medical_visual_processor()
    return visual_chain.invoke({"image": image, "prompt": "Summarize the text from this medical document image. Be comprehensive and precise."})

def create_summary_chain(callback_manager = None):
    summary_llm = ChatAnthropic(
        model='claude-3-5-sonnet-20241022',
        temperature=0,
        top_p=0.7, # reduce randomness
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        callbacks=callback_manager.handlers if callback_manager else None
        )
    
    medical_history_prompt = PromptTemplate.from_template("""
        Create a patient history based on the following medical records. Be comprehensive and precise.
        
        Important requirements:
        1. Include ALL diagnoses with their clinical status
        2. Include ALL dates exactly as they appear
        3. Include ALL test results with their exact values
        4. Maintain strict chronological order
        5. Do not skip any medical events
        
        Medical Records:
        {context}
        
        Previous attempt (if any):
        {previous_generation}

        Previous feedback (if any):
        {feedback}

        Response Format:
        1. Start with "**Diagnosis(es):**" followed by numbered diagnoses in clinical significance order
        2. Each diagnosis must include:
        - Complete condition name
        - Clinical status (active/resolved)
        - Date of diagnosis
        - Any staging information
        3. Then "**History:**" with:
        - Patient demographics (name, age, sex, breed)
        - Referring information
        - Chronological events with exact dates
        - All diagnostic tests and results
        - Current medications with exact dosages
        4. End with any pending items
    """)

    return (
        {
            "context": lambda x: x["context"],
            "previous_generation": lambda x: x.get("previous_generation", ""),
            "feedback": lambda x: x.get("feedback", "")
        }
        | medical_history_prompt
        | summary_llm
        | StrOutputParser()
    )

def create_grade_chain(callback_manager = None):
    grading_llm = ChatAnthropic(
        model='claude-3-5-sonnet-20241022',
        temperature=0,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        callbacks=callback_manager.handlers if callback_manager else None)

    grade_prompt = PromptTemplate.from_template("""
        You are a grader assessing the generated summary of a veterinary medical record for clinical use. 
        Given the medical record, does the generated summary capture all diagnoses, tests, times, dosage amounts, 
        and that they are accurate. If there are any mistakes in the generated summary, it does not meet all conditions. 
        Give a binary 'yes' or 'no' if the document meets all conditions. If `no` explain the reason as to why the 
        document does not meet all conditions.
        
        Medical Record:
        {context}

        Generated Summary:
        {generated_summary}
    """)

    return (
        {
            "context": lambda x: x["context"],
            "generated_summary": lambda x: x["generated_summary"]
        }
        | grade_prompt
        | grading_llm.with_structured_output(GradeSummary)
    )

def create_decomposition_chain(callback_manager = None):
    decomposer_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.8,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        callbacks=callback_manager.handlers if callback_manager else None)
    
    system = """Break down complex queries into simpler component queries that together cover all aspects of a patient's medical history. Create 5-7 specific, searchable queries that focus on:

    1. Primary diagnoses and their clinical status
    2. Chronological progression of symptoms and treatments
    3. Diagnostic test results and their dates
    4. Current medications and dosages
    5. Pending tests or follow-ups
    6. Key patient demographics and referral information
    7. Any complications or secondary conditions

    Each query should be:
    - Focused on a single aspect
    - Specific enough to return relevant results
    - Written in a way that would match the language used in medical records
    
    Format output as a list of specific, searchable queries."""

    validation_prompt = """
    Verify that your queries:
    1. Cover all essential aspects of patient history
    2. Are specific and searchable
    3. Use appropriate medical terminology
    4. Together provide a complete picture
    
    If any aspect is missing, add additional queries to cover it.
    """
    
    decomposition_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {query} \n Break it down into simpler queries."),
        ("system", validation_prompt)
    ])
    
    return decomposition_prompt | decomposer_llm.with_structured_output(QueryDecomposer)

def hyde_expansion(query: str, temperature: float = 0.5) -> str:
    """Expand a query using hypothetical document expansion."""
    hyde_llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=temperature,
        openai_api_key=os.getenv('OPENAI_API_KEY'),)
    hyde_prompt = f"Write a hypothetical passage that would directly answer this query. Be specific and detailed: {query}"
    hyde_doc = hyde_llm.invoke(hyde_prompt)
    return f"{query} {hyde_doc.content}"

def rerank_docs(docs: List, query: str, top_k: Optional[int] = None) -> List:
    """Rerank documents using a cross-encoder model."""
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(scores, docs))
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    reranked = [doc for score, doc in sorted_docs]
    
    return reranked[:top_k] if top_k else reranked

