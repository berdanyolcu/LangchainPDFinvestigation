from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()

# File paths for multiple PDFs
file_paths = [
    "/Users/berdn90s/Downloads/Staj Isler/20170428 Evening.pdf",
    "/Users/berdn90s/Downloads/Staj Isler/20170428 MS Evening.pdf",
    "/Users/berdn90s/Downloads/Staj Isler/20170429 Weekend Summary.pdf"
]

# Load and split all documents
all_docs = []
for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    all_docs.extend(docs)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# Create a vector store for all documents
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits,
    embedding=AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=os.environ["AZURE_EMBEDDING_OPENAI_ENDPOINT"],
        chunk_size=1000
    )
)

# Set up the retriever
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo-16k-1",
    api_version="2024-05-01-preview"
)

# Define the system prompt
system_prompt = (
    """
    You are an assistant for question-answering tasks. Use the retrieved context, including graphs, numerical data, and textual explanations, to answer the question accurately.

    When applicable:
    - Reference trends, patterns, or specific values from the graphs.
    - Compare numerical data to historical benchmarks if requested.
    - Relate graphical data to textual context (e.g., policy, forecasts, or market dynamics).

    For all responses:
    - Provide a concise, three-sentence answer.
    - Mention the document name and page number in your response.
    - If the required data is not available, clearly state: 'The data required to answer this question is not found in the retrieved context.'
    """
    "\n\n"
    "{context}"
)

# Create the question-answer chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# TODO: RATE CRITERIA: 1. Alignment with the Question, 2. Use of Graph and Context, 3. Conciseness, 4.Traceability, 5. Insightfulness


# Define the questions
question_comphrensive = "The USDA biodiesel production forecast is based on current usage trends, while B100 demand fluctuates seasonally. Considering the data provided, how does the producer margin trend impact the viability of US biodiesel production, and how might changes in foreign imports affect this outlook?"
# AI rates this question, answer 4.4/5 by checking pdfs


# TODO: This question is challenging due to it requires numerical comparison (graph vs. text context).
question_Historical_Comparison = "The biodiesel producer margins showed significant fluctuations in the graph. How do the weekly returns for biodiesel plants compare to the 5-year average, and what factors contributed to these changes according to the report?"
# AI rate this questions, answer 4.6/5 by checking pdfs


# TODO: This question is challenging due to Links trends in the graph to future production estimates, Requires interpreting past data to infer near-term forecasts..
question_Predictions = "Based on the graph of biodiesel plant margins, how do the current weekly margins align with historical levels, and what does the data suggest about production figures for March and April?"
# AI rate this answers, 4.8/5 by checking pdfs


# TODO: This question is challenging due to it involves synthesizing graphical trends with textual policy discussion, Requires connecting imports with margin impacts..
question_import_export_dynamics = "The report shows a decline in biodiesel imports from Argentina in earlier months, followed by a sharp rise in April exports according to the graph. How might this trend impact US producer margins, and what policy actions are suggested to counter this?"
# AI rate this answers, 4.7/5 by checking pdfs


# TODO: This question is challenging due to it Combines graphical seasonal data with USDA textual projections. Requires precise extraction of current vs. past year usage figures.
question_seasonal_market_trends = "The graph on biodiesel usage shows seasonal fluctuations in B100 sales and blends. How does the February data compare to the prior year, and what impact might this trend have on achieving USDA’s annual biodiesel forecast?"
# AI rate this answers, 4.7/5 by checking



results = rag_chain.invoke({"input": question_import_export_dynamics})

# Print the output
print("Prompt:")
print(f"Question: {question_import_export_dynamics}")
print(f"Answer: {results['answer']}")

