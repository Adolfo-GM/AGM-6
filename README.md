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
