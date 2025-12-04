# improved_sample_code.py
import os
import httpx
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "sk-GMkhvtGtaMdkp5xVgnk-Vw")

if not API_KEY:
    raise SystemExit("Set GENAI_API_KEY environment variable before running.")

BASE_URL = os.getenv("GENAI_BASE_URL", "https://genailab.tcs.in")
MODEL = os.getenv("GENAI_MODEL", "azure/genailab-maas-gpt-4o")
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() in ("1", "true", "yes")

def main():
    with httpx.Client(verify=False) as client:
        llm = ChatOpenAI(
            base_url=BASE_URL,
            model=MODEL,
            api_key=API_KEY,
            http_client=client,
        )
        try:
            response = llm.invoke("Hi")
            print(response.content)
        except Exception as e:
            print("Request failed:", e)

if __name__ == "__main__":
    main()