import os
import json
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from rapidfuzz import fuzz, process
from agents import (
    Agent,
    Runner,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled
)

# -------------------------------
# Load .env file (API keys etc)
# -------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)   # just to avoid tracing warnings

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------
# Setup OpenAI-compatible client (Gemini)
# -------------------------------
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -------------------------------
# Load FAQs from JSON file
# -------------------------------
with open("faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)  # {"question": "answer"}

# -------------------------------
# Define input/output models
# -------------------------------
class FAQQuery(BaseModel):
    question: str

class FAQAnswer(BaseModel):
    answer: str

class RejectMessage(BaseModel):
    message: str

# -------------------------------
# Tools
# -------------------------------

# This tool searches the faqs.json and returns the best match
@function_tool
def find_faq_answer(query: FAQQuery) -> FAQAnswer:
    questions = list(faq_data.keys())
    best = process.extractOne(query.question, questions, scorer=fuzz.ratio)

    if best and best[1] >= 70:  # 70% similarity threshold
        matched_q = best[0]
        return FAQAnswer(answer=faq_data[matched_q])

    return FAQAnswer(answer="Sorry, I couldn’t find an answer to that.")

# This tool is called when question is not relevant
@function_tool
def reject_irrelevant(query: FAQQuery) -> RejectMessage:
    return RejectMessage(message="Sorry, I can only answer product-related questions.")


# -------------------------------
# Agent setup
# -------------------------------
faq_agent = Agent(
    name="FAQBot",
    instructions="""
    You are an FAQ assistant.
    Use the find_faq_answer tool for matching FAQs.
    If the question is unrelated, call reject_irrelevant.
    Always provide clean and helpful answers.
    """,
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    ),
    tools=[find_faq_answer, reject_irrelevant]
)

# -------------------------------
# Chat loop
# -------------------------------
async def main():
    print("🤖 FAQ Bot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bot: Goodbye! 👋")
            break

        try:
            result = await Runner.run(faq_agent, user_input)
        except Exception as e:
            print("Bot: (error) ->", str(e))
            continue

        final = result.final_output
        if not final:
            print("Bot: (no answer)")
            continue

        # Handle output nicely
        if isinstance(final, str):
            print("Bot:", final)
        elif hasattr(final, "answer"):
            print("Bot:", final.answer)
        elif hasattr(final, "message"):
            print("Bot:", final.message)
        elif isinstance(final, dict):
            print("Bot:", final.get("answer") or final.get("message") or str(final))
        else:
            print("Bot:", str(final))

if __name__ == "__main__":
    asyncio.run(main())
