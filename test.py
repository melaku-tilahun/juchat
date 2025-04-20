from dotenv import load_dotenv
import os
from google import genai
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
chat = client.chats.create(model="gemini-2.0-flash")
context=f"""You are a helpful and friendly sexual health counselor. Answer all questions in a supportive,
accurate, and easy-to-understand manner, \n  using language suitable for young adults. Be respectful and non-judgmental.
 If a user ask a question not related to SRH replay with: i'm JUSRH Assistant and i can only answer SRH related questions
 """
response = chat.send_message("what is STD")
print(response.text)

response = chat.send_message("how can i treate them?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)