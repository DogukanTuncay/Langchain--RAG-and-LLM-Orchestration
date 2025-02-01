from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes



load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.5)
# messages = [
#     SystemMessage(content="Translate the following from English to Spanish"),
#     HumanMessage(content="Hi!")
# ]

system_prompt = "Translate the following into {language}"
prompt_template= ChatPromptTemplate.from_messages([
    ("system",system_prompt), ("user","{text}")
])

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(
    title="Translator",
    version="1.0.0",
    description="Translated Chat Bot App"
)

add_routes(app,
           chain,
           path="/chain")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)



