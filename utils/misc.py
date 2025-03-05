import pyLCIO, glob
import ROOT as rt
from utils.track import Track

def GetFourVector(obj):
    """
    Gives four-momentum in (px,py,pz,E).
    """
    if(type(obj) == Track):
        return obj.GetVector()

    obj_p = obj.getMomentum()
    vec = rt.Math.PxPyPzEVector()
    vec.SetCoordinates(obj_p[0],obj_p[1],obj_p[2], obj.getEnergy())
    return vec

def GetNumEventsTotal(fnames, max_events):
    if(max_events > 1):
        return max_events
    num_events = 0
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.setReadCollectionNames([])

    # ############## LOOP OVER EVENTS AND FILL HISTOGRAMS  #############################
    # Loop over events
    for f in fnames:
        reader.open(f)
        num_events += reader.getNumberOfEvents()
        reader.close()
    return num_events

def ParseInputFiles(input_string):
    try:
        with open(input_string,'r') as f:
            lines = f.readlines()
            files = [x.replace('\n','').strip() for x in lines]
            return files
    except:
        return glob.glob(input_string,recursive=True)
