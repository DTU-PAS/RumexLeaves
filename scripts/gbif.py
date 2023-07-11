import glob
import requests
import os
import io
from PIL import Image

def add_img_metadata(image, line_elements):
    exif = image.getexif()
    exif[0x013B] = line_elements[14]  # TIFF Tag: Artist
    exif[0x010E] = line_elements[5]  # TIFF Tag: Image Descritption
    user_comment = f"Publisher: {line_elements[12]}; Link to image source {line_elements[3]}"
    exif[0x9286] = user_comment  # TIFF Tag: UserComment
    exif[0x8298] = line_elements[-2]  # TIFF Tag: License, todo
    return exif

def main():
    txt_file = "/home/ronja/data/0256024-220831081235567/multimedia.txt"
    img_folder = "/home/ronja/data/0256024-220831081235567/CC_BY_images/"
    multimedia_file = f"{img_folder}/multimedia.txt"
    num_images = 1000000
    start_index = 0

    # Init multimedia.txt if it doesn't exist.
    if not os.path.exists(multimedia_file):
        with open(multimedia_file, "w") as f:
            f.write("File Name; Publisher; Name of Creator; Link to image; License; Title of the image\n")

    # Read gbifs multimedia.txt file
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Keeping track of images, that have already been scraped
    index_keeper = [0]
    img_files = glob.glob(f"{img_folder}/*.png")
    for img_file in img_files:
        img_id = os.path.basename(img_file).split(".")[0]
        index_keeper.append(int(img_id))

    # Scrape images
    count = 0
    for i in range(1, len(lines)):
        # random_index = random.randint(0, len(lines))
        random_index = i
        if random_index in index_keeper or i <= start_index:
            continue
        index_keeper.append(random_index)
        line_elements = lines[random_index].split("\t")
        # remove line breaks
        line_elements = [line_element.replace("\n", "") for line_element in line_elements]

        cc_license = line_elements[-2]
        publisher = line_elements[12]

        # We only want images of a specific license.
        if cc_license != "http://creativecommons.org/licenses/by/4.0/" or publisher not in ["iNaturalist"]:
            continue
        # Skipping botanical images as much as possible, because they are useless and big,
        if line_elements[10] in ["Division of Botany, Yale Peabody Museum", "Conveyor Belt"]:
            continue
        if line_elements[14] in ["University of Tennessee Vascular Herbarium (TENN)\n", "Naturalis Biodiversity Center\n", "Carnegie Museum of Natural History\n","Old Dominion University Herbarium (ODU)\n"]:
            continue

        count += 1
        url = line_elements[3]
        print(f"Downloading {url} with license {cc_license}...")
        try:
            img_data = requests.get(url).content
        except Exception as e:
            print(e)
            continue
        image = Image.open(io.BytesIO(img_data))
        exif = add_img_metadata(image, line_elements)

        image.save(f'{img_folder}/{random_index}.png', exif=exif)
        with open(multimedia_file, "a") as f:
            f.write(f"{random_index}.png; {publisher}; {line_elements[14]}; {url}; {cc_license}; {line_elements[5]}\n")

        if count > num_images:
            break
    print(f"{count} images downloaded.")


if __name__ == "__main__":
    main()
