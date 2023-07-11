import os
import glob
from PIL import Image
from annotation_converter.AnnotationConverter import AnnotationConverter



def add_img_metadata(image, line_elements):
    exif = image.getexif()
    exif[0x013B] = line_elements[14]  # TIFF Tag: Artist
    exif[0x010E] = line_elements[5]  # TIFF Tag: Image Descritption
    user_comment = f"Publisher: {line_elements[12]}; Link to image source {line_elements[3]}"
    exif[0x9286] = user_comment  # TIFF Tag: UserComment
    exif[0x8298] = line_elements[-2]  # TIFF Tag: License, todo
    return exif

def main():
    img_folder = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    out_folder = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist2"
    img_files = glob.glob(f"{img_folder}/*/*.png")
    img_files.extend(glob.glob(f"{img_folder}/*/*.jpg"))
    multimedia_file = f"{out_folder}/references.txt"

    txt_file = F"{img_folder}/multimedia.txt"
    lines = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    os.makedirs(out_folder, exist_ok=True)

    with open(multimedia_file, "w") as f:
        f.write("File Name; Publisher; Name of Creator; Link to image; License; Title of the image\n")

    for img_file in img_files:
        id = int(os.path.basename(img_file).replace(".png", "").replace(".jpg", ""))
        line_elements = lines[id].split("\t")
        line_elements = [line_element.replace("\n", "") for line_element in line_elements]
        cc_license = line_elements[-2]
        publisher = line_elements[12]
        url = line_elements[3]
        with open(multimedia_file, "a") as f:
            f.write(f"{id}; {publisher}; {line_elements[14]}; {url}; {cc_license}; {line_elements[5]}\n")
        img = Image.open(img_file)
        exif = add_img_metadata(img, line_elements)
        img.save(f'{out_folder}/{id}.jpg', exif=exif)
        annotations_file = f"{os.path.dirname(img_file)}/annotations.xml"
        annotation = AnnotationConverter.read_cvat_by_id(annotations_file, os.path.basename(img_file))
        annotation.image_name = f"{id}.jpg"
        AnnotationConverter.extend_cvat(annotation, f"{out_folder}/annotations.xml")



if __name__ == "__main__":
    main()