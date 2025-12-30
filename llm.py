from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os 
from dotenv import load_dotenv
import httpx

load_dotenv()

BASE_URL="https://genailab.tcs.in/"
client = httpx.Client(verify=False)

API_KEY = os.getenv("GENAI_API_KEY")
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# FAST_LLM=ChatOpenAI(
#     base_url=BASE_URL,
#     openai_api_key=API_KEY,
#     model="azure/genailab-maas-gpt-4o",
#     temperature=0.1,
#     http_client=client
# )

FAST_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    client=client,
    temperature=0.3
)

# EMBEDDING_MODEL=OpenAIEmbeddings(
#     base_url=BASE_URL,
#     openai_api_key=API_KEY,
#     model="azure/genailab-maas-text-embedding-3-large",
#     http_client=client
# )

EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=GOOGLE_API_KEY
)

# REASONING_LLM=ChatOpenAI(
#     base_url=BASE_URL,
#     openai_api_key=API_KEY,
#     model="gemini-2.5-pro",
#     http_client=client
# )

REASONING_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    client=client,
    temperature=0.3
)

# print(EMBEDDING_MODEL.embed_query("hi"))
