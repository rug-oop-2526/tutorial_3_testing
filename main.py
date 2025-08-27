import numpy as np

from preprocessing.numeric_cleaner import NumericCleaner

def main():
    data = np.array([
        [4.2, 3.0, 9.5],
        [5.2, np.nan, 3.7],
        [8.3, 9.0, 6.2]
    ])

    cleaner = NumericCleaner.from_defaults()

    print(cleaner(data))

if __name__ == "__main__":
    main()