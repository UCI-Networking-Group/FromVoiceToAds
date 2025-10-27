import json


def print_stats(data_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_categories = set().union(*data.values())
    print(len(all_categories))
    for cate in all_categories:
        print(cate)

    # compute lengths
    lengths = {key: len(val) for key, val in data.items()}

    # find min and max
    min_key = min(lengths, key=lengths.get)
    max_key = max(lengths, key=lengths.get)

    print("Minimum:", min_key, "with", lengths[min_key], "items")
    print("Maximum:", max_key, "with", lengths[max_key], "items")


if __name__ == "__main__":
    print_stats("results/AdvertiserAudiences.json")
    print_stats("results/AmazonAudiences.json")
