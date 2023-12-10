import os

def get_csv_lengths(folder_path):
    with open('length.txt', 'w') as length_file:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(folder_path, filename)
                with open(csv_path, 'r') as csv_file:
                    csv_length = sum(1 for line in csv_file)
                    length_file.write(f"{csv_length}\n")

if __name__ == "__main__":
    folder_path = "data/simulations_20"
    get_csv_lengths(folder_path)

# testing