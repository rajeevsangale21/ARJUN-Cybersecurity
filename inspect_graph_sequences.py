import pickle
import os
import numpy as np

GRAPH_FILE = "data/processed/graph_sequences.pkl"

print("=" * 70)
print("ARJUN GRAPH SEQUENCE FILE INSPECTION")
print("=" * 70)
print()

if not os.path.exists(GRAPH_FILE):
    raise FileNotFoundError(
        f"File not found: {GRAPH_FILE}"
    )

with open(GRAPH_FILE, "rb") as f:
    data = pickle.load(f)

print("Top-level Python type:")
print(type(data))
print()

if isinstance(data, dict):

    print("Dictionary keys:")
    for key in data.keys():
        print(f"  - {key}")

    print()

    for key, value in data.items():

        print("-" * 70)
        print(f"KEY: {key}")
        print(f"TYPE: {type(value)}")

        if hasattr(value, "shape"):
            print(f"SHAPE: {value.shape}")

        try:
            print(f"LENGTH: {len(value)}")
        except TypeError:
            print("LENGTH: N/A")

        print()

elif isinstance(data, (list, tuple)):

    print(f"Top-level length: {len(data)}")
    print()

    for i, item in enumerate(data[:10]):

        print("-" * 70)
        print(f"ITEM {i}")
        print(f"TYPE: {type(item)}")

        if isinstance(item, dict):

            print("DICT KEYS:")
            print(list(item.keys()))

            for key, value in item.items():

                print(
                    f"  {key}: "
                    f"type={type(value)}"
                )

                if hasattr(value, "shape"):
                    print(
                        f"       shape={value.shape}"
                    )

                try:
                    print(
                        f"       length={len(value)}"
                    )
                except TypeError:
                    pass

        elif isinstance(item, (list, tuple)):

            print(
                f"Nested length: {len(item)}"
            )

            if len(item) > 0:

                first = item[0]

                print(
                    f"First nested item type: "
                    f"{type(first)}"
                )

                if isinstance(first, dict):
                    print(
                        "First nested item keys:"
                    )
                    print(
                        list(first.keys())
                    )

        elif isinstance(item, np.ndarray):

            print(f"Array shape: {item.shape}")
            print(f"Array dtype: {item.dtype}")

        print()

else:

    print("Unexpected top-level structure.")
    print(data)

print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)