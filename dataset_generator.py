import random


def get_binary_string(size):
    return "".join(random.choice('01') for _ in range(size))


def save_to_file(rows, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(row + '\n')


def main():
    dataset1 = [get_binary_string(50) for _ in range(100000)]
    save_to_file(dataset1, 'dataset50')

    dataset2 = [get_binary_string(random.randint(1, 50)) for _ in range(100000)]
    save_to_file(dataset2, 'datasetRand')


if __name__ == '__main__':
    main()
