from . import DataGenerator
from . import ModelTrainer
from . import SegmentLoad
from . import WSIPreProcessing
from . import WSILoad
from . import Utilities
from. import MaskMaker

class data(object):

    def __init__(self, slicesDir=None):
        self.slicesDir=slicesDir
        self.annotations=list()

    def addAnnotations(self, annotation):
        self.annotations.append(annotation)
