# RumexLeaves

[Dataset](https://data.dtu.dk/articles/dataset/_strong_RumexLeaves_Dataset_introduced_by_Paper_Fine-grained_Leaf_Analysis_for_Efficient_Weeding_Robots_strong_/23659524)

[Paper](ToDo)

[Github](https://github.com/DTU-PAS/RumexLeaves)


RumexLeaves is a image dataset with fine-grained annotation of Rumex obtusifolius weeds. For each indivudal leaf, we provide a pixel-level segmentation mask as well as a keypoint-guided polyline along stem and major vein of the leaf. The dataset includes 809 images with 7747 annotations, while it is differntiated between two types of datapoints: (1) iNaturalist datapoints have been downloaded from the plant publisher iNaturalist and (2) RoboRumex that have been collected with a Husky Robot platform. Both variants originate from real-world settings. The following table gives an overview of the dataset.


|                | Total | iNaturalist | RoboRumex |
|----------------|-------|-------------|-----------|
| # Images         | 809     | 690           | 119         |
| # Leaves w stem  | 3460     | 3250           | 210         |
| # Leaves wo stem | 4287     | 3916           | 371         |


## Example Images
### iNaturalist Samples
<p float="left">
  <img src="imgs/iNaturalist_samples.png" width="700" />
</p>

### RoboRumex Samples
<p float="left">
  <img src="imgs/RoboRumex_samples.png" width="700" /> 
</p>

## Getting started: Pytorch Dataset Class
Download data
```
wget https://data.dtu.dk/ndownloader/files/41521812
```
Install dependencies
```
pip install -r requirements.txt
```
The Pytorch Datasets allows an easy entrypoint to work with the dataset.
To visualize some example images, please run.
```
python rumex_leaves/visualize_img_data.py --data_folder <path-to-your-extracted-RumexLeaves-folder> --num_images <number-of-images-to-display> --datapoint_type <iNaturalist/RoboRumex>
```

## Citation

If you find this work useful in your research, please cite:
```
@article{fine_grained_2023,
author = {Güldenring, Ronja and Anderse, Rasmus Eckholdt and Nalpantidis, Lazaros},
title = {Fine-grained Leaf Analysis for Efficient Weeding Robots},
journal = {tba},
year = {2023}
}
```