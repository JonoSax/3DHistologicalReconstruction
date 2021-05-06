# 3D Histological reconstructions
Full registration and colour normalisation of histological images. Data collected by Hanna Allerkamp et.al of the UoA.

This programme is not solving a new problem. Existing softwares such as "registration Virtual Stack Slices" in ImageJ (https://imagej.net/Register_Virtual_Stack_Slices) do a fantastic job of registeration (both rigidly and elastically). However this workflow uses the original image for registration which puts a lot of stress on RAM, especially for the ultra high resolution images often used in histology, when the process fails there is no easy method to correct the process and these tools are good at registerting a few sequential sections but for large specimens the cumulative errors (from my observations) result in failure (especially of the data set this was developed for, the Boyd Collection)

This programme is designed to combine the time saving attributes and repeatability of an automatic workflow with the accuracy of manual annotations, specifically to register very large specimens. This can be performed with a sequence of images of any resolution on anything from a laptop to a high performance computer. The only requirement is the ability to run python and launch a GUI.

# Key features 

* Registering any resolution images on everything from a laptop to a high performance computer. (NOTE this is set up to extract from ndpi files but if using standard formats just ensure they are placed in the appropriate location of the created directory)

* Integrating a manual feature identification process if there is failure in the automatic process to achieve gold standard registeration. 

* The outputs of the registration are extracted features which can be easily labelled as either 2D patches or 3D volumes.

# Set up

With Python 3.8.# (tested on 3.8.5) install the requirements

This has been tested on a MacBook Pro, 8gb Ram, 1.7 GHz Quad-Core Intel Core i7 with MacOS 10.15.6 to 11.3.1
NOTE recommend using pyenv to manage python versions: https://github.com/pyenv/pyenv

Next, pip install python requirements

    pyenv install 3.8.5                 # optional
    pip install pip --upgrade           # ensure latest pip version installed
    pip install -r requirements.txt    

0b - OPTIONAL: only needs to be done if extracting from NDPI files, otherwise continue to 1.

First install openslide (https://openslide.org)
On a Mac (with brew) this is: 

    brew install openslide

Install ndpitools

    https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/, extracts the tif images from the ndpi file

**Set the dataHome paths for each script**

Run the main_automaticAnnotations script to perform a linear + NL registration and perform feature extraction. It is automatic unless there is feature finding failure, in which case follow the on screen instructions to select features on the sections.

Run the SegSectionTraining to launch an integrated labelling, 2D model training and segmentation script on the BASELINE images.

Run the main_fullScaleTransform to transform the FULLSCALE images exactly the same as the baseline images

**Parameter settings** 

* Defaults are already pre-set so the only variables that need to change are:

* **dataHome**: to the directory contain the image (ndpi) files
    
* **size**: the resolution to extract from the ndpi files

    * options are 20, 10, 5, 2.5, 1.25, 0.675, **2.5** is from my usage a good compromise.
    
    * lower size = better quality/**MUCH** slower speed and higher RAM usage.
    
    * Every increment is a half in the zoom level (a size of **3**, aka 2.5x resolution, has proven to be a sufficient compromise of speed and resolution).
    
* **res**: the scale to downsample the image to use for feature matching and mask making 
    
    * higher = better quality/slower speeds (**0.2** has proven to be sufficient from experience)
    
    * 1 = same resolution, 0.5 = half size etc.
    
* **cpuNo**: number of cores to use for parallelisations

    * each CPU represents a single process being run on a seperate thread.
    
    * set to 1 for serialisation (allowing debugging).
    
    * On a low memory machine (ie a laptop) set to a lower number, this will take longer but place less demand on your computer. If using a high performance computer set to a higher number to process faster. 
    
    * NOTE it is advised to not use all your CPUs if you want to perform other tasks (ie if you have 8 cores, use 6 so that you can still browse web etc if you want to).
    
    * NOTE for the non-rigid registration, any cpuNo above 1 will result in 100% cpu usage given the parallelisaiton method is different 

* **featsMin (featfind)**: minimum number of matches to identify on between each section 

    * More matches results in a more robust registration process, but also increases the time required for each and increases the possibility of requiring manual feature selections

    * NOTE featsMin for featfind and nonRigidAlign are different

* **featsMin (nonRigidAlign)**:  minimum number of sections a feature has to propogate through to be used as a feature trajectory

    * The more sections a feature propogates through the more likely it is to be significant so higher numbers enforce only "strong" feature being processed. 
    
    * If the number of sections being processed is small then it may result in failures and small features don't always necessarily correlate with insignificant ones... featsMin = 10 is default and has worked fine for me, there are other checks which ensure the features being used are robust. 

* **featsMax**: maximum number of features to use per section

    * If a section has more than this threshold number of features then the "strongest" features are selected (for all sections to be propogated). By default set to 100. 

* **dist**: minimum distance between identified features (in pixels)

    * Minimum distance between features in pixels. Prevents a cluster of features being identifed in a single location enforcing a more even spread of features. Lower numbers results in less overlap of features. Anything less than 30 and overlap is very likely because the feature finding operator (SIFT) uses a 16x16 search grid.

        nonRigidAlign(dataHome, size, cpuNo = cpuNo, \
        featsMin = 10, dist = 30, featsMax = 100, errorThreshold = 200, \
            distFeats = 50, sect = 100, selectCriteria = "length", \
                flowThreshold = 0.05, fixFeatures = False, plot = False)

* **errorThreshold**: the error per feature (pixels^2) threshold for registration

    * Distances between reference and target features allowed during registration. If a transformation doesn't meet this threshold, high error features are removed until a transformation that meets the threshold is found. 
    
    * A lower threshold results in better alignment of the features which are chosen but can also lead to many features being removed so the alignment is for less features. 300 is the default for images which are 6MP, larger images should have a larger error etc.

* **sect**: the proporptional area of the image used for feature tracking and feature extraction

    * A section size of 100 = 1% of the image (50 would = 2% etc.). 

    * A larger section (lower value) results in more of the image being captured during tracking which can result in the tracking of larger features, but during extraction is will lack image specific precision. A smaller section (high value) is the opposite. Default is 100.

* **selectCriteria**: critieria used to determine if a feature is to be selected for NL deformation

    * options are either *length* or *smoothness*

    * length prioritises longer features (ie if it goes through more sections, its inclusion in processing is prioritised), **DEFAULT**

    * smoothness prioritises features which, compared between the smoothed and raw feature trajectories, have less errors

* **flowThreshold**: Maximum diagonal proporption size deformation of the image 

    * for example if a image is 300x400 pixels its diagonal is 500 pixels. For the default (0.05) a maximum NL deformaiton of 25 pixels is allowed.

    * lower values restrict the NL deformation preventing unrealistic deformations, however this results in the features tracked having a lesser ability to deform features. It is *RECOMMENDED* to set this to 1 first (ie all features used in deformation) and if this produces unrealistic deformations then adjust this value as necessary

    * See NLAlignedSamplesSmall-->flowThreshold for the results of deformations to visualise deformation transformations.

* **fixFeatures**: Identify patterns of missing samples

    * If the naming convention of the images corresponds with their locations in a stack, then this can be used to interpolate between missing features (when set to True). 

* **plot**: plots some helpful stuff

    * Most of the useful things are saved but if set to true, helpful plots will appear to help understand/debug. 

* **Labels**: the labels being used for training

    * In segSectionsTraining, specifiy what kind of tissues are being labelled. I would recommend using abbreviated terms for the labelling (ie myometrium = m...) to make the labelling process easier where possible.
    
# Outputs are saved in the following directories:

    └── Size        # all files are processed per extracted scale
        ├── alignedSamples      # linear aligned sections
        │    └── feats              # transformed feature positions
        ├── FeatureSections # stores the feature trajectory position and extracted patches)
        │    └── #ID feat           #True = 3D extraction as it appears in the reconstruction, False = only the patch corresponding is extracted
        ├── fullScale*          # Full scale transformations as related to each section
        ├── images              # base-line slide image
        ├── info                # reference and target section feature positions for linear transformation
        ├── infoNL              # reference and target section feature positions for NL transformation
        ├── maskedSamples       # extract and, if specified, colour normalised sections
        │    ├── masks              # .pbm files used to mask images
        │    └── plots              # graphs showing the bounding and identification of each section
        ├── NLAlignedSamplesSmall # the base-line sections with linear/NL deformations
        │    └── flowMagnidue       # dense deformation matrices
        ├── RealignedSamples    # sections aligned using tracked features)
        │    └── feats              # transformed feature positions
        └── tifFiles            # full-scale images


NOTE on operation
During the base-line creation, there are some hardcoded rotations which were specific to the original data set, ignore these (unless you want to add your own hard coded rotations). It is HIGHLY recommended that you MANUALLY ensure that the full-scale images are at least rotated in the right oriengation and any highly degraded images are deleted, BEFORE processing.

The NL warping can use an Nvidia GPU if available. Ensure that cuda 10.1 is loaded onto your session before running.

If running on a headless machine (such as a remote server), ensure port forwarding is activated as critical GUIs for featuring finding are launched. 

This is designed to work initially with ndpi files, HOWEVER if you are working with the RGB files directly just store your files in a directory corresponding with the "images" directory. Just use a fake size parameter. Ensure that the images being used are .tif format, if not then the images won't be found (just convert to .tif if in another format such as .jpg or .png)