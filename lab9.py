from collections import defaultdict


def shift_word(word, shift):
    result = []
    for char in word:
        if char.isalpha():
            base = ord("A") if char.isupper() else ord("a")
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)
    return "".join(result)


def find_anagram_groups(word_list):
    anagram_dict = defaultdict(list)
    for word in word_list:
        sorted_word = "".join(sorted(word))
        anagram_dict[sorted_word].append(word)
    for group in anagram_dict.values():
        if len(group) > 1:
            print(group)


def most_frequent_letters(text):
    letter_freq = {}
    for letter in text:
        if letter.isalpha():
            letter = letter.lower()
            letter_freq[letter] = letter_freq.get(letter, 0) + 1
    sorted_letters = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
    for letter, freq in sorted_letters:
        print(f"-- {letter}: {freq}", end=" ")


def word_distance(word1, word2):
    if len(word1) != len(word2):
        raise ValueError("Hai từ phải có cùng độ dài.")
    return sum(c1 != c2 for c1, c2 in zip(word1, word2))


def find_metathesis_pairs(word_list):
    def is_metathesis_pair(w1, w2):
        diffs = [(c1, c2) for c1, c2 in zip(w1, w2) if c1 != c2]
        # Kiểm tra nếu có đúng 2 vị trí khác nhau và hoán đổi chúng sẽ tạo thành từ kia
        return len(diffs) == 2 and diffs[0] == diffs[1][::-1]

    # Tạo từ điển nhóm các từ là anagram của nhau
    anagram_dict = defaultdict(list)
    for word in word_list:
        key = "".join(sorted(word))
        anagram_dict[key].append(word)

    # Tìm các cặp metathesis trong từng nhóm anagram
    result = []
    for anagrams in anagram_dict.values():
        for i in range(len(anagrams)):
            for j in range(i + 1, len(anagrams)):
                w1, w2 = anagrams[i], anagrams[j]
                if is_metathesis_pair(w1, w2):
                    result.append((w1, w2))
    return result


# t = (tuple([1, 2]), tuple([3, 4]))
# my_dict = {t: "value"}
# print(my_dict)
# print(shift_word("cheer", 7))  # Kết quả mong đợi: jolly
# print(shift_word("melon", 16))  # Kết quả mong đợi: cubed
# words = [
#     "listen",
#     "silent",
#     "enlist",
#     "inlets",
#     "google",
#     "gogole",
#     "evil",
#     "vile",
#     "veil",
#     "live",
# ]
# find_anagram_groups(words)
# sample_text = "This is a sample English text."
# most_frequent_letters(sample_text)
# print(word_distance("apple", "ammle"))  # Kết quả: 2
# print(word_distance("hello", "hullo"))  # Kết quả: 1
# print(word_distance("test", "tent"))  # Kết quả: 1
# print(word_distance("same", "same"))  # Kết quả: 0
words1 = [
    "converse",
    "conserve",
    "listen",
    "silent",
    "enlist",
    "inlets",
    "google",
    "gogole",
]
pairs = find_metathesis_pairs(words1)
for pair in pairs:
    print(pair)
