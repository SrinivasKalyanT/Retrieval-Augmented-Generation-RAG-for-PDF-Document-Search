from langchain.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain.chains import RetrievalQA # type: ignore
from langchain.llms import HuggingFacePipeline # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.llms import HuggingFacePipeline # type: ignore

def load_tinyllama_pipeline():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Avoid using device_map="auto" if not using accelerate's inference utilities
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = 0 if torch.cuda.is_available() else -1  # device=-1 means CPU
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        return_full_text=False
    )

    return HuggingFacePipeline(pipeline=pipe)


def load_rag_chain(index_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


    llm = load_tinyllama_pipeline()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    return qa_chain
