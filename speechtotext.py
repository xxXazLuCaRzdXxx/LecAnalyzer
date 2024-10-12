import ollama
import json
import whisper

def process_text_with_ollama(input_text, model_name="llama3.1", chunk_size=2000):
    # Split the input text into chunks
    chunks = [input_text[i:i+chunk_size] for i in range(0, len(input_text), chunk_size)]
    
    responses = []
    
    for chunk in chunks:
        prompt = f"""
        You are given text from a part of lecture.
        It contains mix of English and Hindi.

        Your task:
        1. Understand the given text.
        2. Return all the information present inside the text in structured form.
        3. Make sure you are accurate and precise.
        4. Make sure to cover all points. Ensure completeness.
        5. Give only required output.

        Text: {chunk}
        """
        
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        responses.append(response['message']['content'])
    
    return responses

def save_responses(responses, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

model = whisper.load_model("large")

# Transcribe an audio file
result = model.transcribe("path to your audio file",language='en')
print(result['text'])
ollama_responses = process_text_with_ollama(result['text'])
save_responses(ollama_responses, "analysis.json")

print("Ollama analysis completed saved to analysis.json")