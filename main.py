import pandas as pd
import numpy as np
from groq import Groq
import os
import pinecone

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from IPython.display import display, HTML

