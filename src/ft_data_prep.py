import re
import json

def convert_qa_to_squad_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all Q: ... A: ... pairs
    pattern = re.compile(r'Q: (.*?)\nA: (.*?)(?=\nQ: |\Z)', re.DOTALL)
    pairs = pattern.findall(content)

    squad_format = []
    for question, answer in pairs:
        answer = answer.strip().replace('\n', ' ')
        item = {
            'context': answer,
            'question': question.strip(),
            'answers': {
                'text': [answer],
                'answer_start': [0]  # since context = answer, start is 0
            }
        }
        squad_format.append(item)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_format, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    convert_qa_to_squad_format('data/QandA.txt', 'data/processed/QA.json')
    print("Conversion complete. JSON saved to data/processed/QA.json")
