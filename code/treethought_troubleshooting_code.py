import json
import openai
import os
import random
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureOpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from langchain.vectorstores import Chroma

os.environ["AZURE_OPENAI_API_KEY"] = "***"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://**.openai.azure.com/"

OPENAI_DEPLOYMENT_NAME = "gpt_4"
OPENAI_EMBEDDING_MODEL_NAME = "textembeddingada"
MODEL_NAME = "gpt-4"
OPENAI_API_VERSION = "2023-11-01-preview"


# Define the tree structure
tree = {
    "initial_check": {
        "prompt": "Is the issue specific to one device or all devices?",
        "branches": {
            "device-specific": "device_specific_troubleshooting",
            "network-wide": "network_wide_troubleshooting"
        }
    },
    "device_specific_troubleshooting": {
        "prompt": "Please check if your device is on airplane mode or if the network adapter is disabled.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "advanced_device_checks"
        }
    },
    "advanced_device_checks": {
        "prompt": "Try restarting your device or running a network troubleshooter.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "check_browser"
        }
    },
    "network_wide_troubleshooting": {
        "prompt": "Restart your router and check if other devices can connect.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "isp_issue_check"
        }
    },
    "isp_issue_check": {
        "prompt": "Check if there are any outages reported by your ISP.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "contact_isp"
        }
    },
    "check_browser": {
        "prompt": "Open a browser and check if you can visit a website.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "dns_issues"
        }
    },
    "dns_issues": {
        "prompt": "Try changing your DNS settings or flush the DNS cache.",
        "branches": {
            "solved": "confirm_resolution",
            "not solved": "contact_isp"
        }
    },
    "confirm_resolution": {
        "prompt": "Has the issue been resolved?",
        "branches": {
            "yes": "end",
            "no": "contact_isp"
        }
    },
    "contact_isp": {
        "prompt": "You can use internet now, but first restart computer"
    },
    "end": {
        "prompt": "Thank you for using our service."
    }
}

# Save to JSON
with open(r'C:\lsg\重要项目\tranformer_building\troubleshooting_tree.json', 'w') as json_file:
    json.dump(tree, json_file, indent=4)
    

OPENAI_EMBEDDING_MODEL_DEP_NAME = "textembedding"
OPENAI_EMBEDDING_MODEL_NAME = 'text-embedding-ada'

# text emd model name  deployment name   
# textembeddingada002  text-embedding-ada-002
Embeddings_model = AzureOpenAIEmbeddings(deployment = "textembeddingada002",
                   model = "text-embedding-ada-002",
                   azure_endpoint = "https://helpdeskchatbot-gpt4.openai.azure.com/",
                   openai_api_type="azure")

## data Data Injection into Chroma
'''
loader = TextLoader('intertnet_ts.txt', encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents=chunks, embedding = Embeddings_model,
           persist_directory="data/chroma_db")
'''


def get_retriever():
    loaded_vectordb = Chroma(persist_directory = "data/chroma_db", 
                             embedding_function = Embeddings_model)
    retriever = loaded_vectordb.as_retriever(search_type="mmr", k = 5)
    return retriever


# Initialize Azure Chat OpenAI model
chat_model = AzureChatOpenAI(
    openai_api_version=OPENAI_API_VERSION,
    azure_deployment=OPENAI_DEPLOYMENT_NAME,
    temperature=0
)


chat_retriever = get_retriever()

chat_memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    input_key="question",
    output_key='answer',
    return_messages=True
)

# Define templates for human and system messages
system_template = """
 You are an expert for internet service troubleshooting expert, 
 You only answer questions related to the technical questions about internet service. 
 Ignore the personal identifiable information and answer generally. 
 ---------------
 {context}
 """

human_template = """Previous conversation: {chat_history}
     Please provide an answer with less than 150 English words for the following new human question: {question}
     """
 
messages = [
     SystemMessagePromptTemplate.from_template(system_template),
     HumanMessagePromptTemplate.from_template(human_template)
 ]
    

# Initialize the chain
qa_prompt = ChatPromptTemplate.from_messages(messages)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    chain_type='stuff',
    retriever=chat_retriever,
    memory=chat_memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

##try one example to see if LLM is working####
conversation_history = []    
user_request = "my internet is not working, can you solve this"
bot_response = qa_chain({"question": user_request, "chat_history": conversation_history})['answer']

#################start#############################


class InternetTroubleshootingAssistant:
    def __init__(self):
        self.qa_chain = qa_chain
        self.current_node = "initial_check"
        self.history = []  # Initialize the history attribute to store conversation history
        with open('troubleshooting_tree.json', 'r') as json_file:
            self.tree = json.load(json_file)
        self.all_prompts = [node['prompt'] for node in self.tree.values()]  # Load all prompts for other uses
        self.prompt_index = 0  # Initialize an index to track the current prompt in all_prompts    

    def ask_question(self):
        # Check if the current prompt index is within the range of available prompts
        if self.prompt_index < len(self.all_prompts):
            # Display the prompt associated with the current index
            query = f"AI Assistant(query): {self.all_prompts[self.prompt_index]}"
            print(query)
            # Increment the index to point to the next prompt
            self.prompt_index += 1
        else:
            # Reset or handle the end of the prompts list
            print("All troubleshooting steps have been exhausted.")
            self.prompt_index = 0  # Reset the index if you want to loop through the prompts again

    def handle_response(self, user_response):
        print(f"Customer: {user_response}")

        # Determine the branch based on user response
        decision = 'solved' if 'yes' in user_response.lower() else 'not solved'
        if decision in self.tree[self.current_node]['branches']:
            self.current_node = self.tree[self.current_node]['branches'][decision]
        else:
            print("")

        # Generate a response from LLM using PRM after changing the node
        if self.current_node != "end":
            response, score = self.eval_prm(user_response)
            print(f"AI Assistant: Best solution - {response} with score {score}")
            #self.ask_question()  # Ask the next question according to the troubleshooting path
        else:
            print("")

    def eval_prm(self, user_input):
        # Generate multiple simulated responses from the LLM
        context_prompt = f"{self.tree[self.current_node]['prompt']} Customer last said: {user_input}"
        responses = [
            self.qa_chain({"question": context_prompt, "chat_history": self.history}),
            self.qa_chain({"question": context_prompt, "chat_history": self.history}),
            self.qa_chain({"question": context_prompt, "chat_history": self.history})
        ]
        
        # Evaluate each response and calculate scores
        scores = []
        for response in responses:
            # Example scoring mechanism: consider length and randomly simulate context relevance
            length_score = len(response['answer'].split()) / 100
            context_relevance_score = random.uniform(0.5, 1.0)  # Simulated relevance score
            total_score = (length_score + context_relevance_score) / 2
            scores.append(total_score)
        
        # Select the response with the highest score
        max_score_index = scores.index(max(scores))
        best_response = responses[max_score_index]['answer']
        best_score = scores[max_score_index]

        return best_response, best_score

    def user_interaction(self, user_input):
        # Manage the interaction flow
        if self.current_node == "initial_check":
            self.ask_question()  # Start with the initial question on first interaction
        self.handle_response(user_input)

# Example usage
assistant = InternetTroubleshootingAssistant()
# Start the interaction with the initial question
#assistant.ask_question()

user_responses = [
    "yes, it is only my laptop",  # Moves from initial_check to device_specific_troubleshooting
    "no, it is not on airplane mode",  # Moves from device_specific_troubleshooting to advanced_device_checks
    "no, I still have issues after restarting my device or running a network",  # Moves from advanced_device_checks to check_browser
    "no, I cannot connect with phone after restarting my router",  # Moves from check_browser to confirm_resolution
    "no, does not have any outage ",  # Moves from confirm_resolution back to contact_isp
    "no, I cannot visit any website right now",  # Moves from initial_check to network_wide_troubleshooting
    "yes, I  can connect internet after changing your DNS settings",  # Moves from network_wide_troubleshooting to isp_issue_check
    "yes, it is working now",  # Moves from isp_issue_check to contact_isp
    "yes, sure, I will first restart my computer ",  # Already at contact_isp, prompts action
    "yes, Thanks a lot"  # Confirmation and end of troubleshooting
]

for response in user_responses:
    print('------------------------------------')
    assistant.user_interaction(response)






