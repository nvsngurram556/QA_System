#from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
# For fine-tuning, use your processed QA JSON data
import time
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load pretrained DistilBERT QA model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load Exact Match metric for evaluation
exact_match_metric = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")

# Define test samples (question, context, ground truth answer)
test_samples = [
    {
        "question": "What is IRFC’s core business model?",
        "context": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases.",
        "answer": "Borrowing from financial markets to finance rolling stock and railway infrastructure which are then leased to the Ministry of Railways under finance leases."
    },
    {
        "question": "When did IRFC commence project funding to MoR under the finance-lease model?",
        "context": "IRFC commenced project funding to MoR in October 2015; as per a May 23, 2017 MoU with MoR.",
        "answer": "October 2015"
    },
    {
        "question": "How many employees did IRFC have on March 31, 2025?",
        "context": "45 employees; women comprised 20% of the workforce.",
        "answer": "45 employees"
    },
    {
        "question": "What was IRFC’s Revenue from Operations in FY2024-25?",
        "context": "₹27,152.14 crore.",
        "answer": "₹27,152.14 crore"
    },
    {
        "question": "What was Profit After Tax (PAT) in FY2024-25 and growth vs. FY2023-24?",
        "context": "PAT ₹6,502.00 crore, up 1.40% from ₹6,412.11 crore.",
        "answer": "₹6,502.00 crore"
    },
    {
        "question": "What was IRFC’s net worth as on March 31, 2025?",
        "context": "₹52,667.77 crore.",
        "answer": "₹52,667.77 crore"
    },
    {
        "question": "What was the Debt-Equity ratio in FY2024-25?",
        "context": "7.83 times (vs. 8.38 in FY2023-24).",
        "answer": "7.83 times"
    },
    {
        "question": "What were Operating Profit FY2024-25?",
        "context": "Operating Profit 23.93%",
        "answer": "23.93%"
    },
    {
        "question": "Was there any income tax expense in FY2024-25?",
        "context": "zero tax liability due to MAT provisions.",
        "answer": "zero tax liability"
    },
    {
        "question": "What interim dividends did the Board declare in FY2024-25?",
        "context": "Two interim dividends of 8% each (₹0.80 per share) on Nov 4, 2024 (paid Nov 27, 2024) and Mar 17, 2025 (paid Mar 27, 2025).",
        "answer": "Two interim dividends of 8% each (₹0.80 per share)"
    }
]

total_time = 0
correct_em_count = 0
total_samples = len(test_samples)

print("Baseline benchmarking with DistilBERT pre-trained model:")
print("-------------------------------------------------------")

for sample in test_samples:
    start_time = time.time()
    result = qa_pipeline(question=sample["question"], context=sample["context"])
    end_time = time.time()
    
    predicted_answer = result['answer']
    confidence = result.get('score', None)
    inference_time = end_time - start_time
    total_time += inference_time
    
    # Calculate exact match (case-insensitive)
    em = 1 if predicted_answer.strip().lower() == sample["answer"].strip().lower() else 0
    correct_em_count += em
    
    print(f"Q: {sample['question']}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Ground Truth Answer: {sample['answer']}")
    print(f"Confidence Score: {confidence:.4f}")
    print(f"Exact Match: {em}")
    print(f"Inference Time: {inference_time:.4f} seconds\n")

# Summary metrics
print("-------------------------------------------------------")
print(f"Average Exact Match Accuracy: {correct_em_count / total_samples:.2f}")
print(f"Average Inference Time per Question: {total_time / total_samples:.4f} seconds")

