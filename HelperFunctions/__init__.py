# link to all the HelperFunctions

from . import Utilities
from . import SegmentLoad
from. import MaskMaker
from. import WSILoad
from . import WSIExtract
from. import noiseClassGen
from . import DataGenerator
from. import targetTissue
from . import ModelTrainer
from. import ModelEvaluater
from. import SegmentID
from. import stackAligned


class data(object):

    def __init__(self, slicesDir=None):
        self.slicesDir=slicesDir
        self.annotations=list()

    def addAnnotations(self, annotation):
        self.annotations.append(annotation)
