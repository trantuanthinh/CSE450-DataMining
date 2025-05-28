def is_anagram(word1, word2):
    return sorted(word1.lower()) == sorted(word2.lower())


def is_palindrome(word):
    return "".join(reversed(word.lower())) == word.lower()


def reverse_sentence(sentence):
    words = sentence.strip().split()
    reversed_words = words[::-1]
    reversed_words = [word.lower() for word in reversed_words]
    if reversed_words:
        reversed_words[0] = reversed_words[0].capitalize()
    return " ".join(reversed_words)


def total_length(input_file):
    try:
        total = 0
        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                words = line.strip().split()
                total += sum(len(word) for word in words)
        return total
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{input_file}': {e}")
        return None


# Assignment 1
word_list = [
    "stake",
    "skate",
    "teaks",
    "steak",
    "takes",
    "bakes",
    "cakes",
    "keats",
    "speak",
]
target_word = "takes"
anagrams_of_takes = [
    word
    for word in word_list
    if is_anagram(word, target_word) and word.lower() != target_word.lower()
]
print("Anagrams of 'takes':", anagrams_of_takes)

# Assignment 2
print("Is Parlindrome:", is_palindrome("racecar"))
setence = "I enjoy many outdoor activities, e.g., hiking, camping, and fishing."

# Assignment 3
print("Reverse Sentence:", reverse_sentence(setence))

# Assignment 4
print("Total length:", total_length("words.txt"))
