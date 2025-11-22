"""
AI service for essay analysis using LangChain and ZhipuAI.
"""
import os
import logging
from typing import Optional

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException
    # Try different import paths for ChatZhipuAI
    try:
        from langchain.chat_models import ChatZhipuAI
    except ImportError:
        try:
            from langchain_community.chat_models import ChatZhipuAI
        except ImportError:
            from langchain_zhipuai import ChatZhipuAI
except ImportError as e:
    raise ImportError(
        "LangChain dependencies not installed. "
        "Please install: pip install langchain langchain-zhipuai"
    ) from e

from app.schemas import FullReport

# Configure logging
logger = logging.getLogger(__name__)


class AIAnalysisFailedException(Exception):
    """Custom exception for AI analysis failures after retries."""
    pass

# Initialize ZhipuAI model
# Get API key from environment variable
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
if not ZHIPU_API_KEY:
    logger.warning("ZHIPU_API_KEY not found in environment variables")

# Initialize ChatZhipuAI
def _get_chat_model():
    """Initialize ChatZhipuAI model with proper parameters."""
    if not ZHIPU_API_KEY:
        return None
    
    try:
        # Try with zhipuai_api_key parameter
        return ChatZhipuAI(
            zhipuai_api_key=ZHIPU_API_KEY,
            model="glm-4",  # Using GLM-4 model
            temperature=0.7,
        )
    except TypeError:
        # Fallback: try with api_key parameter
        return ChatZhipuAI(
            api_key=ZHIPU_API_KEY,
            model="glm-4",
            temperature=0.7,
        )

chat_model = _get_chat_model()


# Create Pydantic Output Parser
output_parser = PydanticOutputParser(pydantic_object=FullReport)

# Create Prompt Template
PROMPT_TEMPLATE = """你是一位顶级的英文编辑专家，拥有丰富的英语写作和编辑经验。
你的任务是分析用户提供的英文文本，并提供专业的编辑建议和润色版本。

请仔细分析以下文本，并提供：
1. 写作目标分析：分析文本的写作目的、目标受众和整体结构
2. 句子级别分析：逐句分析每句话的问题、修正建议和改进方向
3. 润色版本：提供完整润色后的文本

原始文本：
{original_text}

{context_section}

请严格按照以下JSON格式输出分析结果：
{format_instructions}
"""


def _build_prompt(context: Optional[str] = None) -> ChatPromptTemplate:
    """
    Build the prompt template with context.
    
    Args:
        context: Optional context information about writing goals, audience, etc.
    
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    context_section = ""
    if context:
        context_section = f"\n上下文信息：\n{context}\n"
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt.partial(
        format_instructions=output_parser.get_format_instructions(),
        context_section=context_section,
    )
    return prompt


async def analyze_essay(text: str, context: Optional[str] = None) -> FullReport:
    """
    Analyze English essay using AI and return structured report.
    
    Args:
        text: The original English text to analyze
        context: Optional context information (writing goals, audience, etc.)
    
    Returns:
        FullReport: Structured analysis report
    
    Raises:
        ValueError: If API key is missing or text is empty
        AIAnalysisFailedException: If AI analysis fails after max retries
    """
    if not ZHIPU_API_KEY or not chat_model:
        error_msg = "ZHIPU_API_KEY is not configured"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not text or not text.strip():
        error_msg = "Text cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    max_retries = 3
    last_error = None
    
    logger.info(f"Starting essay analysis for text length: {len(text)}")
    
    # Build prompt (only need to build once)
    prompt = _build_prompt(context)
    chain = prompt | chat_model
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}: Invoking AI model")
            
            # Invoke chain asynchronously
            response = await chain.ainvoke({
                "original_text": text,
            })
            
            logger.info(f"Attempt {attempt}: Received response from AI model")
            
            # Parse output
            try:
                # Get content from response (handle different response types)
                content = response.content if hasattr(response, 'content') else str(response)
                report = output_parser.parse(content)
                logger.info(f"Attempt {attempt}: Successfully parsed AI response into FullReport")
                return report
            
            except OutputParserException as e:
                content = response.content if hasattr(response, 'content') else str(response)
                logger.warning(f"Attempt {attempt}: Failed to parse AI output: {e}")
                logger.warning(f"Attempt {attempt}: Raw AI response: {content}")
                last_error = e
                
                # If this is the last attempt, don't retry
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                    raise AIAnalysisFailedException(
                        f"AI analysis failed after {max_retries} attempts. "
                        f"Last parsing error: {str(e)}. "
                        f"Raw response: {content[:500]}..."  # Truncate long responses
                    ) from e
                
                # Continue to next retry
                continue
            
            except Exception as e:
                content = response.content if hasattr(response, 'content') else str(response)
                logger.warning(f"Attempt {attempt}: Unexpected error during parsing: {e}")
                logger.warning(f"Attempt {attempt}: Raw AI response: {content}")
                last_error = e
                
                # If this is the last attempt, don't retry
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                    raise AIAnalysisFailedException(
                        f"AI analysis failed after {max_retries} attempts. "
                        f"Last parsing error: {str(e)}. "
                        f"Raw response: {content[:500]}..."
                    ) from e
                
                # Continue to next retry
                continue
        
        except Exception as e:
            # Handle errors during chain invocation (not parsing errors)
            logger.warning(f"Attempt {attempt}: Error during AI model invocation: {e}")
            last_error = e
            
            # If this is the last attempt, don't retry
            if attempt == max_retries:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                raise AIAnalysisFailedException(
                    f"AI analysis failed after {max_retries} attempts. "
                    f"Last error: {str(e)}"
                ) from e
            
            # Continue to next retry
            continue
    
    # This should never be reached, but just in case
    raise AIAnalysisFailedException(
        f"AI analysis failed after {max_retries} attempts. "
        f"Last error: {str(last_error) if last_error else 'Unknown error'}"
    )

