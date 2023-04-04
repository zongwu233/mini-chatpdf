from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from llama_index.prompts.prompts import QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
     
import os 
from dotenv import load_dotenv

load_dotenv()

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY == "":
  raise Exception("Need set OPENAI_API_KEY")


WORK_DIR = "."


# Load pdf to documents

from pathlib import Path
from llama_index import download_loader

def load_file(file_path):
  inde_file_name  = file_path + ".index"

  #从本地 PDF 文件中提取文本pdfminer.six，该包用于PyPDF2加载亚洲语言
  # 更多其他类型文件查看 https://llamahub.ai/ 
  CJKPDFReader = download_loader("CJKPDFReader")

  loader = CJKPDFReader()
  index_file = os.path.join(Path(WORK_DIR), Path(inde_file_name))
  # 查看是否已经缓存了这个pdf文件的index
  if os.path.exists(index_file) == False:
    documents = loader.load_data(file=os.path.join(Path(WORK_DIR), Path(file_path)))
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(index_file)
  else:
    index = GPTSimpleVectorIndex.load_from_disk(index_file)
  return index


QUESTION_ANSWER_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "{query_str}\n"
)

QUESTION_ANSWER_PROMPT_TMPL_2 = """
You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Context information is below.
=========
{context_str}
=========
{query_str}
"""

QUESTION_ANSWER_PROMPT = QuestionAnswerPrompt(QUESTION_ANSWER_PROMPT_TMPL_2)


def chat(index,query):
  llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo"))

  result = index.query(
      query,
      llm_predictor=llm_predictor,
      text_qa_template=QUESTION_ANSWER_PROMPT,
      # default: For the given index, “create and refine” an answer by sequentially 
      #   going through each Node; make a separate LLM call per Node. Good for more 
      #   detailed answers.
      # compact: For the given index, “compact” the prompt during each LLM call 
      #   by stuffing as many Node text chunks that can fit within the maximum prompt size. 
      #   If there are too many chunks to stuff in one prompt, “create and refine” an answer 
      #   by going through multiple prompts.
      # tree_summarize: Given a set of Nodes and the query, recursively construct a 
      #   tree and return the root node as the response. Good for summarization purposes.
      response_mode="tree_summarize",
      similarity_top_k=3,
      # mode="default" will a create and refine an answer sequentially through 
      #   the nodes of the list. 
      # mode="embedding" will synthesize an answer by 
      #   fetching the top-k nodes by embedding similarity.
      mode="embedding",
  )
  print(f"Token used: {llm_predictor.last_token_usage}, total used: {llm_predictor.total_tokens_used}")
  return result

# It's not work now, please don't use it.
# Bug: https://github.com/jerryjliu/llama_index/issues/831
def chat_stream(index,query):
  llm_predictor_stream = LLMPredictor(llm=ChatOpenAI(
    temperature=0.2, model_name="gpt-3.5-turbo", stream=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True
    )
  )

  return index.query(
      query,
      llm_predictor=llm_predictor_stream,
      text_qa_template=QUESTION_ANSWER_PROMPT,
      response_mode="tree_summarize",
      similarity_top_k=3,
      streaming=True,
      mode="embedding",
  )


if __name__ == "__main__":
    while True:
      file_path  = input("Enter the path of a book: ")
      if not file_path.endswith(".pdf"):
        print("Please enter the path of a pdf file .")
      else:
        break
    index = load_file(file_path)
    flag = True
    while flag:
      query = input("Enter the question: ")
      if query == "quit"  \
          or query == "exit" \
            or query  == 'q':
        flag = False
      else:
        resp = chat(index,query)
        print(str(resp))
      #print("similar sources:")
      #print(resp.get_formatted_sources())
