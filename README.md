# Segmentation
Full registration and colour normalisation of histological images. Data collected by Hanna Allerkamp et.al of the UoA.

This programme is not solving a new problem. Existing softwares such as "registration Virtual Stack Slices" in ImageJ (https://imagej.net/Register_Virtual_Stack_Slices) do a fantastic job of registeration (both rigidly and elastically). However this workflow uses the original image for registration which puts a lot of stress on RAM, especially for the ultra high resolution images often used in histology, and when the process fails there is no easy method to correct the process.

It is also well accepted that maunually registering images by an expert is the gold standard. However this process is extremely time consuming and subjective between users.

This programme is designed to combine the time saving attributes and repeatability of an automatic workflow with the accuracy of manual annotations. This can be performed with a sequence of images of any resolution on anything from a laptop to a high performance computer. The only requirement is the ability to run python and launch a GUI.

# Key features 

* Registering any resolution images on everything from a laptop to a high performance computer. 

* Integrating a manual feature identification process if there is failure in the automatic process to achieve gold standard registeration. 


# Set up

0 - with Python 3.6.# (tested on 3.6.9) install the requirements

This has been tested on a MacBook Pro, 8gb Ram, 1.7 GHz Quad-Core Intel Core i7 with MacOS 10.15.6
NOTE recommend using pyenv to manage python versions: https://github.com/pyenv/pyenv

First install openslide, on a Mac (with brew) this is

    brew install openslide

Next, pip install python requirements

    pyenv install 3.6.9                 # optional
    pip install pip --upgrade           # ensure latest pip version installed
    pip install -r requirements.txt

Install openslide and ndpitools

    https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/, extracts the tif images from the ndpi file
    
    https://openslide.org/download/, allows properties of the npdi images to be extracted

1 - open the main_automaticAnnotations script

2 - Set the following variables

* **dataHome**: to the directory contain the ndpi files
    
* **size**: the resolution to extract from the ndpi files

    * 0 = highest resolution available (20x zoom for most, 40x zoom if availabe)
    
    * lower size = better quality/**MUCH** slower speed and RAM usage
    
    * Every increment is a half in the zoom level (a size of **3**, aka 2.5x resolution, has proven to be a sufficient compromise of speed and resolution)
    
* **res**: the scale to downsample the image to use for feature matching and mask making 
    
    * higher = better quality/slower speeds (**0.2** has proven to be sufficient from experience)
    
    * 1 = same resolution, 0.5 = half size etc.
    
* **cpuNo**: number of cores to use 

    * each CPU represents a single process being run on a seperate thread
    
    * set to False for serialisation and debugging
    
    * each thread uses a maximum of 2gb of ram when loading tif. On a low memory machine (ie a laptop) set to a lower number, this will take longer but place less demand on your computer. If using a high performance computer set to a higher number to process faster. 
    
    * NOTE it is advised to not use all your CPUs if you want to perform other tasks (ie if you have 8 cores, use 6 so that you can still browse web etc if you want to) 
    
* **features**: from aligned images, the number of features you with to manually extract and align in the highest resoltuion 

3 - Through the featureFinding, aligning and segmentExtraction there are some manual steps (selecting ROI and if the feature finding doesn't work then manual annotations)

# Outputs are saved in the following directories:
* alignedSamples **[FINAL RESULTS]**: stores the images (high and low res) and the features found in their aligned positions 

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

