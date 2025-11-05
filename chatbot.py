import os
import argparse
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from rag.loader import load_and_split_pdfs
from rag.vectorstore import build_or_load_vectorstore
from rag.retrieval import build_multiquery_compressed_retriever
from rag.smalltalk import is_small_talk, small_talk_reply

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the user's question truthfully using only the context below.\n"
        "Do NOT say things like 'based on the context' or 'according to the document'.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="docs", help="Folder with PDF files")
    parser.add_argument("--persist", default="chroma_db", help="Chroma DB directory")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector DB from PDFs")
    parser.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--llm_model", default="llama3", help="Ollama LLM model")
    args = parser.parse_args()

    print("📄 Loading & splitting PDFs...")
    chunks = load_and_split_pdfs(args.docs)

    print(f"🧱 Using persistent Chroma DB at: ./{args.persist}")
    vectorstore = build_or_load_vectorstore(
        chunks=chunks,
        persist_directory=args.persist,
        embedding_model=args.embed_model,
        force_rebuild=args.rebuild
    )

    llm = Ollama(model=args.llm_model)

    retriever = build_multiquery_compressed_retriever(vectorstore, llm)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )

    print("\n✅ Chatbot is ready. Ask anything from your PDF docs.")
    print("   (Type 'exit' to quit)\n")


    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        if query == "":
            print("⚠️  Please ask a valid question.\n")
            continue

        if is_small_talk(query):
            print("Bot:", small_talk_reply(tone="friendly"), "\n")
            continue

        try:
            result = qa_chain.invoke({"query": query})
            print("Bot:", result["result"], "\n")
        except Exception as e:
            print("❌ Error during retrieval/answer:", str(e))
            print("   Tips:")
            print("   • Ensure you ran: `ollama pull {}` and `ollama pull {}`".format(args.embed_model, args.llm_model))
            print("   • If you changed PDFs, try: `python chatbot.py --rebuild`")
            print()

if __name__ == "__main__":
    main()
