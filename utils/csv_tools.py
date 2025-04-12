import csv


def count_csv_rows(filepath: str) -> int:
    """
    Count the number of rows in a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        int: The number of rows in the CSV file.
    """
    count = 0
    with open(filepath, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for _ in reader:
            count += 1
    return count


# Example usage
if __name__ == "__main__":
    path = "../data/historical/XAUUSD_H1_20250412_185050.csv"
    print(f"The CSV file has {count_csv_rows(path)} rows.")
    path = "../data/historical/XAUUSD_H1_processed.csv"
    print(f"The CSV file has {count_csv_rows(path)} rows.")
