from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rag_technique import results, reranked_results, sample_query

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=generator)

def generate_response(query, top_docs, max_tokens=1500):
    # Concatenate passages up to max_tokens limit (approximate by character length)
    max_context_length = 1500  # Approximate max characters to keep within model context
    concatenated_passages = ""
    for doc in top_docs[:2]:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        if len(concatenated_passages) + len(content) + 1 > max_context_length:
            break
        concatenated_passages += content + "\n"

    prompt = f"Answer the question using the context.\nContext: {concatenated_passages}\nQuestion: {query}\nAnswer:"
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"Error: {e}. Please pull the model using 'ollama pull llama2:7b' and try again.")
        response = ""
    return response


# Generate a response using the reranked results
generated_answer = generate_response(sample_query, reranked_results, max_tokens=1500)
print("\nGenerated Answer:")
print(generated_answer)