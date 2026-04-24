
import requests
import json
import random
import re

URI = "http://localhost:11434/api/chat"
MODEL = "llama3.1:latest"

def ai_play(history, model=MODEL, stream=False):
  system_prompt = """You are playing a number guessing game.

Rules:
- Guess a number between 1 and 100
- You will get feedback: low, high, correct

STRICT:
- Reply ONLY with a number
- No words, no explanation
"""

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "History:\n" + "\n".join(history) + "\nNext guess:"}
  ]

  data = {
    "model": MODEL,
    "messages": messages,
    "stream": False,
    "options": {
      "temperature": 0,
      "stop": ["\n"]  # 🔥 important for small models
    }
  }

  response = requests.post(URI, json=data)

  if response.status_code == 200:
    return response.json()["message"]["content"]
  else:
    print("Error:", response.status_code, response.text)
    return None



if __name__ == "__main__":
  number_to_guess = random.randint(1, 100)
  history = []
  attempts = 0

  while True:
    d = ai_play(history)
    

    if not d:
      continue

    attempts += 1

    match = re.search(r"\d+", d)
    if not match:
      print("Invalid output:", d)
      continue

    guess = int(match.group())

    if guess < number_to_guess:
      result = "low"
    elif guess > number_to_guess:
      result = "high"
    else:
      print(f"Correct! {guess} in {attempts} attempts")
      break
    print(f"attempts: {attempts} AI:{guess} result: {result} correct: {number_to_guess}")
    history.append(f"{guess} -> {result}")