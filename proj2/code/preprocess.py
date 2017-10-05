import json
import sys
import string

def convert(raw_file, output_file, vocab_file):
  input_file = open(raw_file, 'r').read().split("\n")
  output = open(output_file, 'w+')
  vocab_file = open(vocab_file, 'w+')

  vocab = set()
  for json_line in input_file:
    try:
      line = json.loads(json_line)
      text = line["body"]
      text = text.lower()
      text = "".join(l for l in text if l not in string.punctuation)
      output.write(text)
    except:
      continue

    vocab.update(text.split())

      
  vocab_file.write("\n".join(list(vocab)))


if __name__ == "__main__":
  convert(sys.argv[1], sys.argv[2], sys.argv[3])

