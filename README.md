# Document Retrieval and Question Answering System

## Overview
This project implements a document retrieval and question-answering system using LangChain and Azure OpenAI models. The goal is to process PDF documents, extract meaningful information, and provide answers to complex questions based on the context derived from the documents. The system uses advanced AI models to handle document embeddings, retrieval, and question generation for dynamic querying of large datasets.

## Key Features
- **Document Loading and Preprocessing**: Utilizes `PyPDFLoader` to load and process multiple PDF files into text.
- **Text Chunking**: Splits large text documents into smaller, more manageable chunks using `RecursiveCharacterTextSplitter`.
- **Vector Store**: Stores document embeddings in an in-memory vector store using `AzureOpenAIEmbeddings`, enabling efficient document retrieval.
- **Question Answering**: Combines document retrieval and language model responses to answer complex questions based on document context using LangChain’s `create_retrieval_chain`.
- **Custom System Prompts**: Customizes AI responses with tailored prompts for accurate, context-aware answers, including handling of graphs, numerical data, and textual explanations.

## Workflow
1. **Loading PDFs**: Multiple PDFs are loaded and processed to extract raw text content.
2. **Text Splitting**: The extracted content is split into smaller chunks to facilitate easier handling by the AI model.
3. **Vectorization**: Text chunks are embedded into a vector store for efficient search and retrieval.
4. **Question-Answering**: The system answers questions by querying the vector store, retrieving relevant document chunks, and generating concise answers based on both graphical and textual data from the documents.

## Example Use Case
The system can be used for answering detailed questions based on complex documents, such as government reports or scientific papers, with considerations of:
- Numerical comparisons (e.g., graphs vs. textual data)
- Seasonal trends and policy impacts
- Historical comparisons and predictions

## Dependencies
- `langchain`
- `langchain-community`
- `langchain-openai`
- `langchain-core`
- `langchain-text-splitters`
- `dotenv`
- `os`
  
## Getting Started
1. Clone the repository and install dependencies.
2. Set up your environment variables for Azure OpenAI credentials in a `.env` file.
3. Place the PDFs you want to analyze in the appropriate file path.
4. Run the script to invoke the question-answering process with your desired queries.
