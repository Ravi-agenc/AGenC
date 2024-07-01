# !pip -q install openai langchain huggingface_hub transformers

# #!pip install --upgrade openai

# #Important to have this
# pip install -U langchain-openai

# !pip install langchain_community

import os

os.environ['OPENAI_API_KEY'] = 'sk-proj-TF5EsPWvvFKneBdAwx3BT3BlbkFJN68YgvsAl0kDrqGP4eAj'


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain

from langchain_openai import ChatOpenAI

#import openai

llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125',
             temperature=0,
             max_tokens = 256)

memory_1 = ConversationBufferMemory()

from langchain.memory import ChatMessageHistory

ConversationChain parameters:
llm --> llm model to use, as instantiated above [reqd]
memory --> default memory store [reqd]
verbose --> if true, intermediate logs will be printed to the console [optional]


conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory_1
)

#conversation.predict(input="How are you today?")

#conversation.predict(input="I am doing good. I had a question about the recipe for ravioli?")

# memory_2 = ConversationSummaryBufferMemory(llm=llm, chat_memory=ChatMessageHistory(memory_1))
# conversation_contd = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory = memory_2
# )

Currently --> Threshold is holding tokens, not the number of conversations

Update --> Tracks number of conversations and updates memory storage based on that

threshold = 5
conv_counter = 0

def switch_memory_if_needed(conversation_chain, input_text, conv_counter):
    output = conversation_chain.predict(input=input_text)
    conversation_length = len(conversation_chain.memory.buffer)

    # Check if the length exceeds the threshold
    # if conversation_length > threshold:
    if (conv_counter == threshold+1): #& (conv_counter > threshold)
        # Create ConversationSummaryBufferMemory and transfer the history
        """ Change max_token_limit from 40 to ~150 to retain more messages before summarizing """
        memory_2 = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40, chat_memory=ChatMessageHistory(messages=memory_1.chat_memory.messages))
        # Update the conversation chain to use the new memory
        conversation_chain.memory = memory_2
        print("Switched to ConversationSummaryBufferMemory")

    return output


def handle_conversation():
    global conv_counter
    print("Chatbot is ready. Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ")
        conv_counter+=1
        if user_input.lower() == 'exit':
            """ Add a line that invokes the function again to summarize the previous conversation and save the entire summarized conversation to the buffer
            """
            break
        response = switch_memory_if_needed(conversation, user_input, conv_counter)
        print("Ad-vice AI:", response)

handle_conversation()







