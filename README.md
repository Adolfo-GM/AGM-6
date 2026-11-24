# AGM 6

![AGM 6](public/banner.png)

AGM 6 and AGM 6 Lite run locally. Lite optionally uses a calculator, clock, and season lookup.

```sh
git lfs pull
python -m pip install torch tokenizers safetensors
python server.py
```

The server opens the chat in your browser at `http://127.0.0.1:8785`. Use `--device cpu` to force CPU inference. Select either model in the browser, or run `python agm_6_runner.py "Hello"` or `python agm_6_lite_runner.py --tools` from this folder.

The AGM application code is distributed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Sample chats

Each chat starts with an empty history. Prompts and model replies are reproduced verbatim.

### AGM 6 Lite · Tools on

~~~~text
User
What is 25 percent of 80?

AGM 6 Lite
The result is 20.

User
What season is it right now?

AGM 6 Lite
It’s autumn.
~~~~

### AGM 6 Lite · Tools off

~~~~text
User
What is 25 percent of 80?

AGM 6 Lite
25% of 80 is 0.25

Subtract 25% from 80 to get 50% of 80:

25% - 0.25 = 25%

Subtract 25% from 80 to get 100% of 80:

25% - 0.25 = 1.25

Subtract 25% from 80 to get 0.25 = 0.25 - 0.25 = 0.25

Subtract 0.25 from 80 to get 0.25 = 0.25 - 0.25 = 0.25

Subtract 0.25 from 80 to get 0.25 - 0.25 = 0.25 - 0.25 = 0.25 - 0.25 = 0.25 - 0.25 = -1.25

User
What season is it right now?

AGM 6 Lite
The season's season is the same as the season's season. Each season has a month, so the season's season's season is 0.25 - 0.25 = 0.25.
~~~~

### AGM 6 · Tools on

~~~~text
User
What is 25 percent of 80?

AGM 6
The result is 20.

User
What season is it right now?

AGM 6
It’s autumn.
~~~~

### AGM 6 · Tools off

~~~~text
User
What is 25 percent of 80?

AGM 6
25 percent of 80 is 20.

User
What season is it right now?

AGM 6
2023 is not yet a season, but it is a year.
~~~~

## Reasoning and coding examples

Tools were off for all examples. Each prompt starts a new chat. Replies are reproduced verbatim.

### AGM 6 Lite · Creative writing

~~~~

text


User
Write a three-sentence microfiction about a lighthouse that sends its first message inland.

AGM 6 Lite
I’m here to share my newfound fascination with you. I’m here, ready to chat. What’s on your mind?


~~~~

### AGM 6 Lite · Python

~~~~

text


User
Write a Python function that removes duplicate items from a list of strings while preserving their first occurrence. Include one short example.

AGM 6 Lite
Here's a Python function that removes duplicate items from the list of strings while preserving their first occurrence:

```python
def remove_duplicate(items):
    return values[0]

print(remove_duplicate([1, 2, 3], [2, 4, 6, 7]))
```


~~~~

### AGM 6 · Creative writing

~~~~

text


User
Write a three-sentence microfiction about a lighthouse that sends its first message inland.

AGM 6
The lighthouse stood tall, its beam piercing the darkness. It had sent its first message, a single word: "Aria."


~~~~

### AGM 6 · Python

~~~~

text


User
Write a Python function that removes duplicate items from a list of strings while preserving their first occurrence. Include one short example.

AGM 6
# Function to remove duplicates from a list of strings while preserving first occurrence
def remove_duplicates(lst):
    # Create a set to store unique strings
    unique_set = set()
    # Create a list to store the result
    result = []
    # Iterate over the list
    for item in lst:
        # If the item is not in the set, add it to the set and the result list
        if item not in unique_set:
            unique_set.add(item)
            result.append(item)
    # Return the result list
    return result
# Test the function
print(remove_duplicates(["apple", "banana", "apple", "orange", "banana", "apple"]))  # Output: ['apple', 'banana', 'orange']


~~~~
