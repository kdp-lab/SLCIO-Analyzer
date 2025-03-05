import numpy as np
import ROOT as rt

def GetTrackFourVector(track, Bfield):
    theta = np.pi/2- np.arctan(track.getTanLambda())
    phi = track.getPhi()
    eta = -np.log(np.tan(theta/2))
    pt  = 0.3 * Bfield / np.abs(track.getOmega() * 1000.)
    track_vec = rt.Math.PtEtaPhiMVector()
    track_vec.SetCoordinates(pt, eta, phi, 0.)
    return track_vec

class Track():
    def __init__(self, lcio_track, Bfield):
        self.vec = GetTrackFourVector(lcio_track,Bfield)
        self.d0 = lcio_track.getD0()
        self.z0 = lcio_track.getZ0()
        self.chi2 = lcio_track.getChi2()
        self.ndf = lcio_track.getNdf()
        self.nhits = lcio_track.getTrackerHits().size()

    def SetD0(self,val):
        self.d0 = val

    def SetZ0(self,val):
        self.z0 = val

    def SetChi2(self,val):
        self.chi2 = val

    def SetNDF(self,val):
        self.ndf = val

    def SetNHits(self,val):
        self.nhits = val

    def GetVector(self):
        return self.vec

    def GetD0(self):
        return self.d0

    def GetZ0(self):
        return self.z0

    def GetChi2(self):
        return self.chi2

    def GetNDF(self):
        return self.ndf

    def GetNHits(self):
        return self.nhits
