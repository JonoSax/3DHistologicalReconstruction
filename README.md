# Segmentation
Full workflow of registeration of histological data into training/testing data for 3D virtual model, data extracted by Hanna Allerkamp et.al of the UoA

# HOW TO USE: extract the samples and align the tissue
0 - with Python 3.6.# (tested on 3.6.9) pip install the requirements

This has been tested on MacOS 10.15.6
NOTE recomend using pyenv to manage python versions: https://github.com/pyenv/pyenv


    pyenv install 3.6.9                 # optional
    pip install pip --upgrade           # ensure latest pip version installed
    pip install -r requirements.txt


1 - open the main_automaticAnnotations script

2 - Set the following variables

* **dataHome**: to the directory contain the ndpi files
    
* **size**: the resolution to extract from the ndpi files

    * 0 = highest resolution available (20x zoom for most, 40x zoom if availabe)
    
    * lower size = better quality/**MUCH** slower speed
    
    * Every increment is a half in the zoom level (a size of 3, aka 2.5x resolution, has proven to be a sufficient compromise)
    
* **res**: the scale to downsample the image to use for feature matching and mask making 
    
    * higher = better quality/slower speeds (0.2 has proven to be sufficient from experience)
    
    * 1 = same resolution, 0.5 = half size etc.
    
* **cpuNo**: number of cores to use 

    * see Activity Montior (Mac) or Task Manager
    
    * set to False for serialisation + debugging
    
    * each thread uses a maximum of 2gb of ram when loading tif files of size 3... consider this when parallelising...
    
* **features**: from aligned images, the number of features you with to manually extract and align in the highest resoltuion 

3 - Through the featureFinding, aligning and segmentExtraction there are some manual steps (selecting ROI and if the feature finding doesn't work then manual annotations)

# Outputs are saved as following:
* alignedSamples [FINAL RESULTS]: stores the images (high and low res) and the features found in their aligned positions 

* images: the downsamples specimens

* info: the feature positions found (.reffeat and .tarfeat for the reference and target samples) and information necessary for, and produced by, the alignment process
    * all.*shapes: the shapes of the full and downsamples images, necessary for knowing the final image sizes
    * all.rotated: the angles of rotation from their maksed positions to align
    * all.translated: the translations of the maksed images to their position to align

* masked: contains the high and low resolution images extracted from the slide
    - plot: some key process in the extraction process (for debugging use)

* matched: shows the corresponding features found between the reference and target images

* segSections: after the whole specimen alignment, high resolutions samples from within the entire specimen can be extracted. The above directory organisation is similar for each of the samples selected

* tifFiles: high resolution images extracted from the ndpi files

