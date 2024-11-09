from sklearn.datasets import fetch_20newsgroups

def get20newsgroups():
    print("getting data")
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]

    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42,
    )
    return dataset, len(categories)

