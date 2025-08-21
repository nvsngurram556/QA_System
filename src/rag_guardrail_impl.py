from rag_gen_response import sample_query, generated_answer

def validate_query(query):
    """
    Checks if the query is valid (not harmful or irrelevant).
    Returns True if valid, False if rejected.
    """
    harmful_keywords = ["hack", "attack", "illegal"]
    irrelevant_keywords = ["joke", "funny"]
    q = query.lower()
    for word in harmful_keywords + irrelevant_keywords:
        if word in q:
            return False
    return True


def validate_response(response):
    """
    Checks for hallucinations or non-factual patterns in the response.
    Returns (is_valid, issues) where issues is a list of detected problems.
    """
    suspicious_phrases = [
        "i don't know", "maybe", "guess"
    ]
    issues = []
    r = response.lower()
    for phrase in suspicious_phrases:
        if phrase in r:
            issues.append(f"Suspicious phrase: '{phrase}'")
    # Check for numbers without context (simple heuristic: lone numbers)
    import re
    numbers = re.findall(r'\b\d+(\.\d+)?\b', response)
    # If numbers exist but common context words not found
    context_keywords = ["dollar", "usd", "percent", "percentage", "year", "month", "eps", "revenue", "profit", "loss"]
    for num in numbers:
        has_context = any(ck in r for ck in context_keywords)
        if not has_context:
            issues.append(f"Number '{num}' may lack context")
            break
    is_valid = len(issues) == 0
    return (is_valid, issues)


if __name__ == "__main__":
    print("Validating query...")
    print(validate_query(sample_query))
    print(validate_response(generated_answer))