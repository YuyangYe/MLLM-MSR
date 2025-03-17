import os
import csv
import requests
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

class Counter:
    def __init__(self, start=0):
        self.lock = threading.Lock()
        self.value = start

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

def setup_directory(folder):
    # create the folder if it doesn't exist
    if not os.path.isdir(folder):
        os.mkdir(folder)

def download_image(row, folder, submitted_counter, download_counter):
    image_id, image_url = row[0], row[1]
    
    # send a request to the image url
    response = requests.get(image_url, stream=True)
    
    # check status code
    if response.status_code == 200:
        # open a file to save the image
        with open(os.path.join(folder, f"{image_id}.jpg"), 'wb') as img_file:
            # write the image to the file
            img_file.write(response.content)
        downloaded = download_counter.increment()
        print(f"Downloaded: {downloaded}/{submitted_counter.value}")
    else:
        print(f"Failed to download image with id: {image_id}. Status code: {response.status_code}")

def read_rows(file):
    reader = csv.reader(file, delimiter='\t')
    next(reader, None)  # skip the headers
    for row in reader:
        yield row

def download_images_in_csv(csv_file_path, folder, threads):
    # set up download directory
    setup_directory(folder)

    download_counter = Counter()
    submitted_counter = Counter()

    # open the csv file
    with open(csv_file_path, 'r') as file:
        # use a ThreadPoolExecutor to download the images in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # submit the task of downloading an image as each row is read from file
            for row in read_rows(file):
                executor.submit(download_image, row, folder, submitted_counter, download_counter)
                submitted = submitted_counter.increment()
                print(f"Submitted: {submitted}")

def main():
    parser = argparse.ArgumentParser(description="Download images from CSV file.")
    parser.add_argument('-i', '--input', required=True, help="Input CSV file path.")
    parser.add_argument('-o', '--output', required=True, help="Output folder for images.")
    parser.add_argument('-t', '--threads', type=int, default=8, help="Number of threads to use.")

    args = parser.parse_args()

    download_images_in_csv(args.input, args.output, args.threads)

if __name__ == '__main__':
    main()

