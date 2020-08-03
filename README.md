# Segmentation
Full workflow of segmentation of histological data into training/testing data for ML model for identifying the arcuate and radial vessels previously identified by Hanna Allerkamp et.al of the UoA

Helper functions are classed into three sections. The below explains what 
each functions purpose in the workflow is:

    - PR_[], Processing the Raw info 

        - SegmentLoad, FULLY FUNCTIONAL
            Converts the raw xml file information from the annotations 
            into information which corresponds to the pixel locations for any 
            given resolution of tif file being processed

        - WSILoad, FULLY FUNCTIONAL 
            Extracts a tif file of the specified zoom level
            
    - SP_[], Sample Processing to get the raw information into a usable form

        - MaskMaker, MOSTLY FUNCTIONAL
            Blood vessels are annotated on the ndpi image via ndpa files. From these
            annotations, identify every pixel on the specified resolution image which
            is an annotated blood vessel

        - SampleFinder, PARTIALLY FUNCTIONAL/UNDER CONSTRUCTION
            Allows a user to select a ROI on a single slide which will create the 
            the necessary positional information to be fed into SegmentExtraction 
            to propogate a feature through an entire stack of tissue

        - SegmentID, CONTINUOUS IMPROVEMENT
            From annotations on the image, align the samples

        - SpecimenID, UNDER CONSTRUCTION
            Extracts the sample exactly from the whole slice and automatically 
            identifies features which can be used by SegmentID to align the samples

    - CI_[], Creating Information which is value add to the tissue

        - DataGenerator
        - ModelEvaluator
        - ModelTrainer
        - NoiseClassGen

        - SegmentExtraction, MOSTLY FUNCTIONAL
            Propogate a selected feature through an entire stack of aligned tissue
            and extract those segment in each slice

        - StackAligned, NON-FUNCTIONAL/UNDER CONSTRUCTION
            From the aligned samples, create a 3D stack of tif images which can be 
            used to create a 3D volume

        - WSIExtract, FULLY FUNCTIONAL
            From the masks identified, extract only the annotated blood vessel 


