from doctest import run_docstring_examples


def uses_none(string, forbidden_letters):
    """
    >>> uses_none("banana", "axyz")
    True
    """
    for letter in string.lower():
        if letter in forbidden_letters.lower():
            return False
    return True


def uses_only(string, available):
    for letter in string.lower():
        if letter not in available.lower():
            return False
    return True


def uses_all(string, required):
    # for letter in required.lower():
    #     if letter not in string.lower():
    #         return False
    # return True
    return uses_only(required, string)


def check_word(word, available, required):
    if len(word) >= 4 and uses_only(word, available) and uses_all(word, required):
        return True
    return False


def score_word(word, available):
    if len(word) == 4 and uses_only(word, available):
        return 1
    elif uses_all(word, available):
        return len(word) + 7
    elif uses_only(word, available):
        return len(word)
    return 0


run_docstring_examples(uses_none, globals())

print(uses_none("banana", "xyz"))
print(uses_none("apple", "efg"))

print(uses_only("banana", "ban"))
print(uses_only("apple", "apl"))

print(uses_all("banana", "ban"))
print(uses_all("apple", "api"))

print(check_word("color", "ACDLORT", "R"))
print(check_word("ratatat", "ACDLORT", "R"))
print(check_word("rat", "ACDLORT", "R"))
print(check_word("told", "ACDLORT", "R"))
print(check_word("bee", "ACDLORT", "R"))

print(score_word("card", "ACDLORT"))
print(score_word("color", "ACDLORT"))
print(score_word("cartload", "ACDLORT"))
