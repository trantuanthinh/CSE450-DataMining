def value_counts(s):
    counts = {}
    for letter in s:
        counts[letter] = counts.get(letter, 0) + 1
    return counts


def has_duplicates(word):
    return len(word) != len(set(word))


def find_repeats(counter):
    return [key for key, count in counter.items() if count > 1]


def add_counters(dict1, dict2):
    combined = {}
    for key in set(dict1) | set(dict2):
        combined[key] = dict1.get(key, 0) + dict2.get(key, 0)
    return combined


def is_interlocking(word, word_set):
    word1 = word[::2]  # Characters at even indices
    word2 = word[1::2]  # Characters at odd indices
    if word1 in word_set and word2 in word_set:
        return True
    return False


text = "brontosaurus"
text1 = "apatosaurus"
counter = value_counts(text)
counter1 = value_counts(text1)
print(counter)
print(counter1)
# print(has_duplicates("hello"))
# print(has_duplicates("world"))
# repeats = find_repeats(counter)
# print("Repeated letters:", repeats)
# print("Add Counter:", add_counters(counter, counter1))
print(
    "Interlocking words:",
    is_interlocking("schooled", {"shoe", "cold"}),
)
