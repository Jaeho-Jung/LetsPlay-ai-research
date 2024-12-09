import os


from openai import OpenAI


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

class GPTService:
    _initialized = False

    def __init__(self, api_key: str):
        """
        Initialize the GPTService class. This ensures the GPT client is only initialized once.
        
        Args:
            api_key (str): API key for authenticating with the OpenAI service.
        """
        if not GPTService._initialized:
            try:
                # log.info("Starting GPT service...")
                self.client = OpenAI(api_key=api_key)
                GPTService._initialized = True
                # log.info("GPT service started successfully!")
            except Exception as e:
                # log.error(f"Error during GPT service initialization: {str(e)}")
                raise

    def get_chat_completion(self, messages, model="gpt-3.5-turbo-instruct"):
            """ 
            Retrieves a chat completion response from the language model using the provided messages.

            Args:
                messages (list): A list of dictionaries representing the conversation, each containing
                                fields such as 'role' and 'content'.
                model (str): The model to be used for generating the response. Default is 'gpt-4o'.

            Returns:
                str: The generated response content from the model.

            Raises:
                RuntimeError: If an error occurs while attempting to get the completion.
            """
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                # log.error(f"Error in fetching chat completion: {str(e)}")
                raise RuntimeError("Failed to fetch chat completion.") from e

api_key = os.getenv('OPENAI_API_KEY')
gpt_service = GPTService(api_key)

user_input = '나는 강아지, 너는 고양이 해!'
context.append({"role": "user", "content": f"{user_input}"})


response = gpt_service.get_chat_completion(context)

print(response)