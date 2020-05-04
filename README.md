# RegEx2DFA
## Setup
```
$ conda env create -f environment.yml
$ conda activate tc
```

## Files
In `regex.txt` you can specify on the first line the regex, using only `(`, `)`, `|` and `*`. All other characters are considered part of the alphabet.

In `words.txt` you can specify a list of words, each on a new line, to test the LambdaNFA / DFA.

In `output.txt` you can see the result of the parsing of each word.

Run `automata.py` (tested with Python 3.8), which will build a Lambda-NFA, then transform it into a DFA. You can see the two drawn in `LambdaNFA.png` and `DFA.png`.
