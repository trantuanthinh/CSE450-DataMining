import re


def head(input_file, num_lines, output_file=None):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            lines = []
            for _ in range(num_lines):
                line = infile.readline()
                if not line:
                    break
                lines.append(line)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{input_file}': {e}")
        return

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines)
        except Exception as e:
            print(f"An error occurred while writing to '{output_file}': {e}")
    else:
        for line in lines:
            print(line, end="")


def check_word(word, target_word):
    if len(word) != 5:
        return "Only 5 letters allowed!"

    if word == target_word:
        return "Correct!"

    result = ["R"] * 5
    for i in range(5):
        if word[i] == target_word[i]:
            result[i] = "G"
        elif word[i] in target_word:
            result[i] = "Y"
    return "".join(result)


# Assignment 1
head("words.txt", 5)

# Assignment 2
# Assignment 3: The target word is "TOTEM"
target_word = "MOWER"
for i in range(0, 6):
    guess = input("Enter your guess: ")
    temp = check_word(guess.upper(), target_word)
    print(temp)
    if temp == "Correct!":
        break

# Assignment 4
pattern = r"\b(?:pale(?:s|d|ness)?|pallor)\b"
text = "Her face turned pale. He pales in comparison. They paled at the sight. The paleness of his skin was evident. The pallor of her complexion was alarming."
matches = re.findall(pattern, text, flags=re.IGNORECASE)
print(f"Total matches: {len(matches)}")
print("Matches found:", matches)
