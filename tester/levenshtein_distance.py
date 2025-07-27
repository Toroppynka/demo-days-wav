import Levenshtein
def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)


if __name__ == "__main__":
    s1 = "1234567890"
    s2 = "0123456789"
    print(f"Расстояние Левенштейна между '{s1}' и '{s2}':", levenshtein_distance(s1, s2))