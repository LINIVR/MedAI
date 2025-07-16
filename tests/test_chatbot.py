import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Chatbot")))

from vectorstorebuilder import build_vectorstore, get_vectorstore
from retrievalqa import get_retrieval_chain





def test_vectorstorebuilder():
    print("Testing Vector Store...")
    try:
        build_vectorstore()
        vs = get_vectorstore()
        assert vs is not None
        print(" Vector Store test passed.\n")
    except Exception as e:
        print(" Vector Store test failed:", e)

def test_retrievalqa():
    print("Testing Retrieval QA Chain...")
    try:
        chain = get_retrieval_chain()
        assert chain is not None
        print("Retrieval QA Chain test passed.\n")
    except Exception as e:
        print(" Retrieval QA Chain test failed:", e)

if __name__ == "__main__":
    test_vectorstorebuilder()
    test_retrievalqa()
