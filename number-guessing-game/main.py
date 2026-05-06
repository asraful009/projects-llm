
import requests
import json
import random
import re

URI = "http://localhost:11434/api/chat"
MODEL = "llama3.1:latest"

def ai_play(low=0, high=100, history=None, model=MODEL, stream=False):
  system_prompt = f"""You are playing a number guessing game.

The valid range is ALWAYS between {low} and {high}.
You MUST stay within this range.

After each guess:
- "low" means your guess is too low → next guess must be higher
- "high" means your guess is too high → next guess must be lower
- "correct" means you are done

STRICT RULES:
- Output ONLY a number
- No words, no explanation
- NEVER go outside the range
- NEVER repeat a previous guess

MANDATORY STRATEGY:
- First guess = middle of range
- After each step, update your internal range using feedback
- Always guess near the middle of the updated range
- Reduce your search space every step
"""

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "History:\n" + "\n".join(history) + "\nNext your number:"}
  ]

  data = {
    "model": MODEL,
    "messages": messages,
    "stream": False,
    "options": {
      "temperature": .51,
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
  low = 1
  high = 100
  number_to_guess = random.randint(low, high)
  history = []
  attempts = 0

  while True:
    d = ai_play(low=low, high=high, history=history)
    

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
    history.append(f"{guess} -> {result}")
    print(f"attempts: {attempts} AI:{guess} result: {result} correct: {number_to_guess} --->> {history}")
