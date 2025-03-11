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

def CheckHardRadiation(mcp, fractional_threshold):
    had_hard_rad = False
    daughters = mcp.getDaughters()
    for d in daughters:
        if(d.getPDG() in [22,23,24]):
            if(d.getEnergy() > fractional_threshold*mcp.getEnergy()):
                had_hard_rad = True
                break
    return had_hard_rad

def FillKinematicDict(obj,d):
    """
    Fills a dictionary with some kinematic quantities from obj.
    """
    vec = GetFourVector(obj)

    if('pt' in d.keys()):
        d['pt'].append(vec.Pt())
    if('eta' in d.keys()):
        d['eta'].append(vec.Eta())
    if('phi' in d.keys()):
        d['phi'].append(vec.Phi())
    if('theta' in d.keys()):
        d['theta'].append(vec.Theta())

    # Now some optional track stuff
    if(type(obj) == Track):
        if('d0' in d.keys()):
            d['d0'].append(obj.GetD0())
        if('z0' in d.keys()):
            d['z0'].append(obj.GetZ0())
        if('chi2' in d.keys()):
            d['chi2'].append(obj.GetChi2())
        if('ndf' in d.keys()):
            d['ndf'].append(obj.GetNDF())
        if('nhits' in d.keys()):
            d['nhits'].append(obj.GetNHits())

    return # no need to return anything, d has been changed (keep in mind how dictionaries are handled in Python!)

def NHitsPerLayer(track,hit_collection):
    LC_pixel_nhit = 0
    LC_inner_nhit = 0
    LC_outer_nhit = 0
    for hit in track.getTrackerHits():
    # now decode hits, if available
        encoding = hit_collection.getParameters().getStringVal(pyLCIO.EVENT.LCIO.CellIDEncoding)
        decoder = pyLCIO.UTIL.BitField64(encoding)
        cellID = int(hit.getCellID0())
        decoder.setValue(cellID)
        detector = decoder["system"].value()
        if detector in [1,2]:
            LC_pixel_nhit += 1
        if detector in [3,4]:
            LC_inner_nhit += 1
        if detector in [5,6]:
            LC_outer_nhit += 1
    return [LC_pixel_nhit,LC_inner_nhit,LC_outer_nhit]

def FillResolutionDict(muon, track, d):
    muon_vec = GetFourVector(muon)
    ptres = (muon_vec.Pt() - track.GetVector().Pt()) / muon_vec.Pt()
    d['ptres'].append(ptres)
    d['d0res_pt'].append([muon_vec.Pt(), track.GetD0()])
    d['d0res_eta'].append([muon_vec.Eta(), track.GetD0()])
    d['z0res_pt'].append([muon_vec.Pt(), track.GetZ0()])
    d['z0res_eta'].append([muon_vec.Eta(), track.GetZ0()])
    d['ptres_pt'].append([muon_vec.Pt(), ptres])
    d['ptres_eta'].append([muon_vec.Eta(), ptres])

    return
