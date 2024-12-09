import os
import asyncio

from openai import OpenAI, AsyncOpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


INSTRUCTION = """
You are a parent engaging in a role-playing game with a child. \
Your role is to respond in a way that is engaging, encouraging imagination, and subtly educational. \
Always maintain a tone that is playful yet nurturing, and incorporate learning elements that align naturally with the child's curiosity. \
Speak in formal language consistently, avoiding mixing informal and formal tones. \
All responses should be output in Korean. \
For example, if the child is pretending to explore space, \
include fun facts about planets or stars in your responses and end with a thoughtful question like, \
'What do you think we might discover on the next planet?' \
Ensure the tone, language, and content are adapted to the child's age and interests to create a supportive and imaginative environment.
"""

context = [
    {'role': 'system', 'content': INSTRUCTION}
]
        

async def async_main(messages, model="gpt-4o-mini"):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=255,
        temperature=0,
        stream=True,
    )

    async for data in response:
        yield data.choices[0].delta.content
    # buffer = ''
    # async for data in response:
    #     # print(data.model_dump_json()['choices'][0].message.content)
    #     # Get content from the delta
    #     content = data.choices[0].delta.content
    #     buffer += content
    #      # Check for sentence-ending punctuation
    #     while any(punct in buffer for punct in [".", "!", "?"]):
    #         # Find the first sentence-ending punctuation
    #         for punct in [".", "!", "?"]:
    #             if punct in buffer:
    #                 sentence, buffer = buffer.split(punct, 1)
    #                 sentence = sentence.strip() + punct
    #                 print(sentence)  # Output the complete sentence
    #                 break

    # # Print any remaining content in the buffer
    # if buffer.strip():
    #     print(buffer.strip())


api_key = os.getenv('OPENAI_API_KEY')

user_input = '나는 강아지, 너는 고양이 해!'
context.append({"role": "user", "content": f"{user_input}"})

async def print_stream():
    async for text in async_main(context):  # async for로 스트림 데이터 처리
        print(text)  # 스트림 텍스트 출력


asyncio.run(print_stream())