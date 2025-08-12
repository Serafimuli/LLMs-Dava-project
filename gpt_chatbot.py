from openai import OpenAI
from retriver import search_books
from tools import get_summary_by_title
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def chat_with_user(user_question: str):
    # Step 1: Use retriever to find the book
    result = search_books(user_question)
    if not result:
        return "No recommendation found for this request."

    recommended_title = result['title']
    short_summary = result['summary']

    # Step 2: Create conversational GPT response
    system_prompt = "You are an assistant that recommends books based on the user's interests."
    user_prompt = (
        f"The user wants a book related to: '{user_question}'. "
        f"I recommend the book: '{recommended_title}'. "
        f"Short summary: {short_summary}. "
        f"Make a conversational recommendation and then provide additional details."
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8
    )
    response = completion.choices[0].message.content

    # Step 3: Tool call (get full summary)
    full_summary = get_summary_by_title(recommended_title)

    return f"{response}\n\n📘 Full summary:\n{full_summary}"
