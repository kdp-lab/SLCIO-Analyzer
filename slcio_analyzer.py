# Based on code written by Leo Rozanov (UChicago)

import sys,glob,json
import pyLCIO
import ROOT as rt
import argparse as ap
import numpy as np

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


def check_hard_radiation(mcp, fractional_threshold):
    had_hard_rad = False
    daughters = mcp.getDaughters()
    for d in daughters:
        if(d.getPDG() in [22,23,24]):
            if(d.getEnergy() > fractional_threshold*mcp.getEnergy()):
                had_hard_rad = True
                break
    return had_hard_rad

def ParseInputFiles(input_string):
    try:
        with open(input_string,'r') as f:
            lines = f.readlines()
            files = [x.replace('\n','').strip() for x in lines]
            return files
    except:
        return glob.glob(input_string,recursive=True)

def GetFourVector(obj):
    """
    Gives four-momentum in (px,py,pz,E).
    """
    if(type(obj) == Track):
        return obj.GetVector()

    obj_p = obj.getMomentum()
    vec = rt.Math.PxPyPzEVector()
    vec.SetCoordinates(*obj_p[0:3], obj.getEnergy())
    return obj

def GetTrackFourVector(track, Bfield):
    theta = np.pi/2- np.arctan(track.getTanLambda())
    phi = track.getPhi()
    eta = -np.log(np.tan(theta/2))
    pt  = 0.3 * Bfield / np.abs(track.getOmega() * 1000.)
    track_vec = rt.Math.PtEtaPhiMVector()
    track_vec.SetCoordinates(pt, eta, phi, 0.)
    return track_vec

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


def main(args):
    rt.gROOT.SetBatch()

    parser = ap.ArgumentParser()
    parser.add_argument('-i','--inputFile',type=str,help='Input text file listing input files, or a glob-compatible string.',required=True)
    parser.add_argument('-n','--nEvents',type=int,default=-1)
    parser.add_argument('-v','--verbose',type=int,default=0)
    parser.add_argument('-o','--outputFile',type=str,default='slcio_analyser.json')
    args = vars(parser.parse_args())

    fnames = ParseInputFiles(args['inputFile'])
    max_events = args['nEvents'] # Set to -1 to run over all events
    verbose = args['verbose'] > 0
    output_json = args['outputFile']

    # Define a bunch of constants used later
    min_dr = 0.005
    Bfield = 5 #T, 3.57 for legacy
    fractional_threshold = 0. # for checking hardness of radiation from muon
    track_pt_min = 0.5 # GeV, I think -Jan
    bad_pt_res_threshold = 25

    # ############## CREATE EMPTY HISTOGRAM OBJECTS  #############################
    # Set up histograms
    binning = {}
    binning["pt"] =  (20,0,2000)
    binning["eta"] = (50,-5,5)
    binning["phi"] = (20,-3.5,3.5)
    binning["n"] = (20,0,20)
    binning["d0_res"] = (100,-0.1,0.1)
    binning["z0_res"] = (100,-0.1,0.1)
    binning['nhits'] = (50,0,50)
    binning['pt_res_hits'] = (100,-0.5,0.5)
    hists = {}

    per_object_vars = ['pt','eta','phi','n']

    for obj in ["pfo", "pfo_mu", "mcp", "mcp_mu", "mcp_mu_match"]:
        for var in per_object_vars:
            key = '{}_{}'.format(obj,var)
            hists[key] = rt.TH1F('h_{}'.format(key),';{};Counts'.format(key), *binning[var])
    for var in binning.keys():
        if var in per_object_vars: continue
        hists[var] = rt.TH1F(var, var, *binning[var])

    # Adding 2D histograms
    hists_2d = {}
    variables_2d = {
        "d0_res_vs_pt": {'binning':(100,0,2000,100,-0.1,0.1),
                        "ylabel": "D0 Resolution", "xlabel": "pT"},
        "d0_res_vs_eta": {'binning':(100,-3,3,100,-0.1,0.1),
                        "ylabel": "D0 Resolution", "xlabel": "Eta"},
        "z0_res_vs_pt": {'binning':(100,0,2000,100,-0.1,0.1),
                        "ylabel": "Z0 Resolution", "xlabel": "pT"},
        "z0_res_vs_eta": {'binning':(100,-3,3,100,-0.1,0.1),
                        "ylabel": "Z0 Resolution", "xlabel": "Eta"},
        "pt_res_vs_eta": {'binning':(100,-3,3,100,-0.5,0.5),
                        "ylabel": "pT Resolution", "xlabel": "Eta"},
        "pt_res_vs_pt": {'binning':(100,0,2000,100,-0.5,0.5),
                        "ylabel": "pT Resolution", "xlabel": "pT"},
    }

    for var in variables_2d:
        hists_2d[var] = rt.TH2F(var, var, *variables_2d[var]["binning"])

    # Making a separate set of binning conventions for plots showing resolutions
    # these plots will all be filled with the difference between a pfo and a mcp object value
    dvariables = {}
    dvariables["dpt"] =     (100,-500,500)
    dvariables["drelpt"] =  (100,-0.5,0.5)
    dvariables["dphi"] =    (100,-0.001,0.001)
    dvariables["deta"] =    (100,-0.001,0.001)
    for obj in ["d_mu"]:
        for var in dvariables:
            key = '{}_{}'.format(obj,var)
            hists[key] = rt.TH1F(key,key, *dvariables[var])

    # Finally making one 2D histogram; this is what I'll use for a pT resolution vs. pT plot.
    h_2d_relpt = rt.TH2F("h_2d_relpt", "h_2d_relpt", *binning['pt'], 500, -0.5, 0.5)

    # Create empty lists for each variable
    mcp_pt = [] #mcp = MCParticle (truth)
    mcp_phi = []
    mcp_eta = []

    pfo_pt = [] #pfo = Particle FLow Object (reconstructed)
    pfo_phi = []
    pfo_eta = []
    pfo_mu_pt = []
    pfo_mu_phi = []
    pfo_mu_eta = []

    mcp_mu_pt = [] #mcp_mu = MCParticle muon
    mcp_mu_phi = []
    mcp_mu_eta = []

    mcp_mu_match_pt = [] #mcp_mu_match = MCParticle muon that was matched to a PFO muon
    mcp_mu_match_phi = []
    mcp_mu_match_eta = []

    d_mu_dpt = [] #d_mu = difference between PFO muon and MCParticle muon
    d_mu_drelpt = []
    d_mu_dphi = []
    d_mu_deta = []

    # TRACK
    d0_res = [] #d0_res = d0 resolution
    track_d0 = [] #d0_res_match = d0 resolution for matched muons
    z0_res = [] #z0_res = z0 resolution
    track_z0 = [] #z0_res_match = z0 resolution for matched muons

    nhits = []
    pixel_nhits = []
    inner_nhits = []
    outer_nhits = []
    pt_res_hits = []

    d0_res_vs_pt = []
    d0_res_vs_eta = []
    z0_res_vs_pt = []
    z0_res_vs_eta = []
    pt_res_vs_eta = [] #track muon pt resolution vs pt
    pt_res_vs_pt = []
    pt_res = []

    # Truth matched
    track_pt = [] #This is track pt
    track_eta = [] #This is track eta
    track_phi = [] #This is track pt
    track_theta = [] #This is track pt
    track_ndf = []
    track_chi2 = []

    pt_match = [] #This is truth pt
    eta_match = [] #This is truth eta
    phi_match = []
    theta_match = []

    # LC Relation track
    LC_pt_match = []
    LC_track_pt = []
    LC_track_eta = []
    LC_eta_match = []
    LC_track_theta = []
    LC_phi_match = []
    LC_ndf = []
    LC_chi2 = []
    LC_d0 = []
    LC_z0 = []
    LC_nhits = []
    LC_pixel_nhits = []
    LC_inner_nhits = []
    LC_outer_nhits = []
    LC_pt_res = []
    LC_dr = []

    # Fake Tracks
    fake_pt = []
    fake_theta = []
    fake_eta = []
    fake_phi = []
    fake_d0 = []
    fake_z0 = []
    fake_ndf = []
    fake_chi2 = []
    fake_nhits = []
    fake_pixel_nhits = []
    fake_inner_nhits = []
    fake_outer_nhits = []

    h2d_relpt = [] #pfo muon pt resolution vs pt

    # no_inner_hits = 0
    event_counter = 0
    num_matched_tracks = 0
    num_dupes = 0
    num_fake_tracks = 0
    # hard_rad_discard = 0
    # total_n_pfo_mu = 0

    collection_names = [
        "MCParticle",
        "PandoraPFOs",
        "SiTracks",
        "SiTracks_Refitted",
        "MCParticle_SiTracks",
        "MCParticle_SiTracks_Refitted",
    ]

    track_collection_name = 'SiTracks_Refitted' # TODO: eventually make this toggleable
    relation_collection_name = 'MCParticle_{}'.format(track_collection_name)
    assert(track_collection_name in collection_names)
    assert(relation_collection_name in collection_names)

    # Treat the hit collections separately -- these might not be present.
    # hit_collection_names = [
    #     "IBTrackerHits",
    #     "IETrackerHits",
    #     "OBTrackerHits",
    #     "OETrackerHits",
    #     "VBTrackerHits",
    #     "VETrackerHits"
    # ]
    hit_collection_names = [
        "VBTrackerHitsConed",
        "VETrackerHitsConed"
    ]

    hit_collection_mask = {key:True for key in hit_collection_names}

    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    try:
        reader.setReadCollectionNames(collection_names + hit_collection_names)
    except:
        reader.setReadCollectionNames(collection_names)

    # ############## LOOP OVER EVENTS AND FILL HISTOGRAMS  #############################
    # Loop over events
    for f in fnames:
        if max_events > 0 and event_counter >= max_events: break

        reader.open(f)
        for i,event in enumerate(reader):
            if max_events > 0 and event_counter >= max_events: break

            if event_counter%100 == 0: print("Processing event %i."%event_counter)

            # Get the collections we care about
            relation_collection = event.getCollection(relation_collection_name)
            relation = pyLCIO.UTIL.LCRelationNavigator(relation_collection)

            mcp_collection = event.getCollection("MCParticle")
            pfo_collection = event.getCollection("PandoraPFOs")
            track_collection = event.getCollection(track_collection_name)

            hit_collections = []
            for hname in hit_collection_names:
                if(not hit_collection_mask[hname]):
                    continue
                try:
                    hit_collections.append(event.getCollection(hname))
                except:
                    hit_collection_mask[hname] = False
                    if(verbose):
                        print('\tDid not find hit collection: {}. Disabling...'.format(hname))
                    pass

            # Make counter variables
            n_mcp_mu = 0
            n_pfo_mu = 0
            has_pfo_mu = False
            pfo_mu_vec = 0

            # Pfos
            pfo_dict = {
                'pt':[],
                'eta':[],
                'phi':[]
            }

            pfo_mu_dict = {
                'pt':[],
                'eta':[],
                'phi':[]
            }

            # Truth-level particles (a.k.a. "MCP" = Monte Carlo particle)
            mcp_dict = {
                'pt':[],
                'eta':[],
                'phi':[]
            }

            # Truth-level muons
            mcp_mu_dict = {
                'pt':[],
                'eta':[],
                'phi':[]
            }

            # MCP that was matched to PFO.
            # TODO: Check that this matching is really working,
            # this dict is filled in a MCP loop if some condition was met
            # in a previous, separate PFO loop.
            mcp_mu_match_dict = {
                'pt':[],
                'eta':[],
                'phi':[]
            }

            # note this dictionary's structure is not like many of the others
            d_mu_dict = {
                'dpt':[],
                'deta':[],
                'dphi':[],
                'relpt':[],
                'pt_relpt':[]
            }

            # Tracks
            matched_track_dict = {
                'pt':[],
                'eta':[],
                'phi':[],
                'theta':[],
                'd0':[],
                'z0':[],
                'ndf':[],
                'chi2':[],
                'nhits':[]
            }

            # Track-matched truth-level muon dict
            matched_muon_dict = {
                'pt':[],
                'eta':[],
                'phi':[],
                'theta':[],
            }

            resolution_dict = {
                'd0res':[],
                'z0res':[],
                'ptres':[],
                'd0res_pt':[],
                'd0res_eta':[],
                'z0res_pt':[],
                'z0res_eta':[],
                'ptres_pt':[],
                'ptres_eta':[]
            }

            # id0_res_match = []
            # iz0_res_match = []

            # Fake Tracks
            fake_track_dict = {
                'pt':[],
                'eta':[],
                'phi':[],
                'theta':[],
                'd0':[],
                'z0':[],
                'ndf':[],
                'chi2':[],
                'nhits':[]
            }

            pfo_mu_index = None
            mcp_muon_index = None
            has_fake_tracks = False

            ##################################################################
            # Loop over the reconstructed objects and fill histograms
            for j,pfo in enumerate(pfo_collection):

                FillKinematicDict(pfo,pfo_dict)

                # hists["pfo_pt"].Fill(pfo_tlv.Perp())
                # hists["pfo_eta"].Fill(pfo_tlv.Eta())
                # hists["pfo_phi"].Fill(pfo_tlv.Phi())


                if abs(pfo.getType())==13:
                    # hists["pfo_mu_pt"].Fill(pfo_tlv.Perp())
                    # hists["pfo_mu_eta"].Fill(pfo_tlv.Eta())
                    # hists["pfo_mu_phi"].Fill(pfo_tlv.Phi())

                    FillKinematicDict(pfo,pfo_mu_dict)
                    # NOTE: This this only keeps the last instance of pfo_mu. Probably OK, since we're using muon gun samples -- but good to keep in mind. -Jan
                    pfo_mu_index = j # keep track of this - will use in track loop later below.
                    n_pfo_mu += 1
                    has_pfo_mu = True
            ##################################################################

            ##################################################################
            # Loop over the truth objects and fill histograms
            for j,mcp in enumerate(mcp_collection):
                FillKinematicDict(mcp,mcp_dict)

                # hists["mcp_pt"].Fill(mcp_tlv.Perp())
                # hists["mcp_eta"].Fill(mcp_tlv.Eta())
                # hists["mcp_phi"].Fill(mcp_tlv.Phi())

                if(abs(mcp.getPDG())==13 and mcp.getGeneratorStatus()==1):

                    # Check if the muon radiated significant energy
                    hard_rad = check_hard_radiation(mcp, fractional_threshold)

                    # hists["mcp_mu_pt"].Fill(mcp_tlv.Perp())
                    # hists["mcp_mu_eta"].Fill(mcp_tlv.Eta())
                    # hists["mcp_mu_phi"].Fill(mcp_tlv.Phi())

                    FillKinematicDict(mcp,mcp_mu_dict)
                    mcp_muon_index = j
                    n_mcp_mu += 1

                    # print("Truth pt, eta, phi:", mcp_tlv.Perp(), mcp_tlv.Eta(), mcp_tlv.Phi())

                    mcp_vec = GetFourVector(mcp)
                    if(mcp_vec.Pt() > track_pt_min): # Remove ultra-low pt tracks
                        tracks = relation.getRelatedToObjects(mcp)

                        for k,track in enumerate(tracks):
                            # theta = np.pi/2- np.arctan(track.getTanLambda())
                            # phi = track.getPhi()
                            # eta = -np.log(np.tan(theta/2))
                            # pt  = 0.3 * Bfield / np.abs(track.getOmega() * 1000.)
                            # track_tlv = rt.TLorentzVector()
                            # track_tlv.SetPtEtaPhiE(pt, eta, phi, 0)

                            track_container = Track(track,Bfield)

                            track_vec = track_container.GetVector()
                            dr = rt.Math.VectorUtil.DeltaR(track_vec,mcp_vec)
                            ptres = (mcp_vec.Pt() - track_vec.Pt()) / mcp_vec.Pt()

                            # TODO: Why the extra nesting in lists of length 1? -Jan
                            LC_track_pt.append([track_vec.Pt()])
                            LC_track_eta.append([track_vec.Eta()])

                            LC_pt_match.append([mcp_vec.Pt()])
                            LC_eta_match.append([mcp_vec.Eta()])

                            LC_track_theta.append([track_vec.Theta()]) # theta = np.pi/2- np.arctan(track.getTanLambda())
                            LC_phi_match.append([track_vec.Phi()])
                            LC_ndf.append([track.getNdf()])
                            LC_chi2.append([track.getChi2()])
                            LC_d0.append([track.getD0()])
                            LC_z0.append([track.getZ0()])
                            LC_nhits.append([track.getTrackerHits().size()])
                            LC_pt_res.append([ptres])
                            LC_dr.append([dr])

                            if(len(hit_collections) > 0):
                                LC_pixel_nhit, LC_inner_nhit, LC_outer_nhit = NHitsPerLayer(track,hit_collections[0])
                                LC_pixel_nhits.append([LC_pixel_nhit])
                                LC_inner_nhits.append([LC_inner_nhit])
                                LC_outer_nhits.append([LC_outer_nhit])

                            num_matched_tracks += 1
                            if hard_rad: # TODO: Not sure this is correct? Doesn't look like anything is being discarded. -Jan
                                hard_rad_discard += 1

                        # For events in which a PFO mu was reconstructed, fill histograms that will
                        # be used for efficiency. Both numerator and denominator must be filled with truth values!
                        # Also fill resolution histograms
                        if has_pfo_mu:
                            # hists["mcp_mu_match_pt"].Fill(mcp_tlv.Perp())
                            # hists["mcp_mu_match_eta"].Fill(mcp_tlv.Eta())
                            # hists["mcp_mu_match_phi"].Fill(mcp_tlv.Phi())

                            FillKinematicDict(mcp, mcp_mu_match_dict)

                            pfo_mu_vec = GetFourVector(pfo_collection[pfo_mu_index])
                            # hists["d_mu_dpt"].Fill(pfo_mu_vec.Perp() - mcp_tlv.Perp())
                            # hists["d_mu_drelpt"].Fill((pfo_mu_vec.Perp() - mcp_tlv.Perp())/mcp_tlv.Perp())
                            # hists["d_mu_deta"].Fill(pfo_mu_vec.Eta() - mcp_tlv.Eta())
                            # hists["d_mu_dphi"].Fill(pfo_mu_vec.Phi() - mcp_tlv.Phi())
                            # h_2d_relpt.Fill(mcp_tlv.Perp(), (pfo_mu_vec.Perp() - mcp_tlv.Perp())/mcp_tlv.Perp())
                            d_mu_dict['dpt'].append(pfo_mu_vec.Pt() - mcp_vec.Pt())
                            d_mu_dict['drelpt'].append((pfo_mu_vec.Pt() - mcp_vec.Pt())/mcp_vec.Pt())
                            d_mu_dict['deta'].append(pfo_mu_vec.Eta() - mcp_vec.Eta())
                            d_mu_dict['dphi'].append(pfo_mu_vec.Phi() - mcp_vec.Phi())
                            d_mu_dict['pt_relpt'].append([mcp_vec.Pt(), (pfo_mu_vec.Pt() - mcp_vec.Pt())/mcp_vec.Pt()])
            ##################################################################

            if(n_mcp_mu > 1):
                print('\tWarning: Found {} truth-level muons in event! Skipping...'.format(n_mcp_mu))

            ##################################################################
            # Loop over the track objects and fill histograms for D0, Z0, and hit counts
            # TODO: I don't entirely understand why some parts of this loop exist. Didn't we already loop over tracks within the loop above? -Jan
            counter = 0
            max_hits = 0
            best_track = None
            for j,track in enumerate(track_collection):

                track_container = Track(track,Bfield)

                # Get the deltaR between each track and the truth muon.
                # TODO: This is notably different than in the old code, which effectively
                #       checked dR against mcp_vec (which would be whatever was the last
                #       truth particle from the above loops!).
                dr = rt.Math.VectorUtil.DeltaR(track_container.GetVector(),mcp_vec)

                # Fake tracks
                if(len(relation.getRelatedFromObjects(track)) == 0): # If there's no associated truth muon
                    has_fake_tracks = True
                    FillKinematicDict(track_container,fake_track_dict)

                    fake_pixel_nhit = 0
                    fake_inner_nhit = 0
                    fake_outer_nhit = 0
                    if(len(hit_collections) > 0):
                        fake_pixel_nhit, fake_inner_nhit, fake_outer_nhit = NHitsPerLayer(track,hit_collections[0])
                        fake_pixel_nhits.append(fake_pixel_nhit)
                        fake_inner_nhits.append(fake_inner_nhit)
                        fake_outer_nhits.append(fake_outer_nhit)
                    num_fake_tracks += 1

                if dr < min_dr: # Do dR check first, then do nhits check
                    if track_container.GetNHits() > max_hits:
                        max_hits = track_container.GetNHits()
                        best_track = track_container
                    counter += 1
                    if counter > 1:
                        num_dupes += 1
                        # print("More than one track in event! # of dupes:", num_dupes)
            ##################################################################################

            # Now compute some resolution stuff, using the track most closely dR-matched to
            if best_track is not None:

                # hists["d0_res"].Fill(d0) # TODO: This seems wrong, d0_res filled with just d0? Similar issue below.-Jan
                # hists["z0_res"].Fill(z0)
                # hists["nhits"].Fill(max_hits)

                # for k, particle_pt in enumerate(imcp_mu_pt): # NOTE: Removing this loop since we will assume only 1 truth-level muon (and skip if otherwise).
                # Maybe can add this back in later. -Jan

                muon_vec = GetFourVector(mcp_collection[mcp_muon_index])
                ptres = (muon_vec.Pt() - best_track.GetVector().Pt()) / muon_vec.Pt()

                # # Fill 2D histograms
                # hists_2d["d0_res_vs_pt"].Fill(particle_pt, d0)
                # hists_2d["d0_res_vs_eta"].Fill(particle_eta, d0)
                # hists_2d["z0_res_vs_pt"].Fill(particle_pt, z0)
                # hists_2d["z0_res_vs_eta"].Fill(particle_eta, z0)
                # hists_2d["pt_res_vs_eta"].Fill(particle_eta, ptres)
                # hists_2d["pt_res_vs_pt"].Fill(particle_pt, ptres)

                #num_matched_tracks += 1

                FillKinematicDict(best_track,matched_track_dict)
                FillKinematicDict(mcp_collection[mcp_muon_index],matched_muon_dict)
                FillResolutionDict(mcp_collection[mcp_muon_index], best_track, resolution_dict)

                # if np.abs(ptres) > bad_pt_res_threshold:
                #     with open("bad_res.txt", "a") as textfile:
                #         textfile.write("Filename: {}\tEvent: {}\tpt_res: {}\n".format(f,event.getEventNumber(),ptres))

                pixel_nhit, inner_nhit, outer_nhit = NHitsPerLayer(track,hit_collections[0])

            ##################################################################

            #print("End of tracks")
            # This is here to check that we never reconstruct multiple muons
            # If we did, we'd have to match the correct muon to the MCP object to do eff/res plots
            # But since we don't, we can skip that step
            if n_pfo_mu > 1:
                print('\tWarning: Found {} reconstructed muons.'.format(n_pfo_mu))

            # hists["mcp_n"].Fill(len(mcpCollection))
            # hists["pfo_n"].Fill(len(pfoCollection))
            # hists["mcp_mu_n"].Fill(n_mcp_mu)
            # hists["pfo_mu_n"].Fill(n_pfo_mu)
            # hists["mcp_mu_match_n"].Fill(n_pfo_mu)


            # Now fill a bunch of things.

            # Truth-level particles.
            mcp_pt.append(mcp_dict['pt'])
            mcp_eta.append(mcp_dict['eta'])
            mcp_phi.append(mcp_dict['phi'])

            # Truth-level muons.
            if(not hard_rad):
                # filling with truth-level muon
                mcp_mu_pt.append(mcp_mu_dict['pt'])
                mcp_mu_eta.append(mcp_mu_dict['eta'])
                mcp_mu_phi.append(mcp_mu_dict['phi'])

                # filling with mu-matched pfo
                if(has_pfo_mu):
                    mcp_mu_match_pt.append(mcp_mu_match_dict['pt'])
                    mcp_mu_match_eta.append(mcp_mu_match_dict['eta'])
                    mcp_mu_match_phi.append(mcp_mu_match_dict['phi'])

                    d_mu_dpt.append(d_mu_dict['dpt'])
                    d_mu_drelpt.append(d_mu_dict['drelpt'])
                    d_mu_deta.append(d_mu_dict['deta'])
                    d_mu_dphi.append(d_mu_dict['dphi'])
                    h2d_relpt.append(d_mu_dict['pt_relpt'])

            # If available, fill information on resolution as well as the matched track
            if(best_track is not None):
                # d0_res.append(id0_res)
                # z0_res.append(iz0_res)
                # nhits.append(inhits)

                # TODO: these might need fixing
                pixel_nhits.append([pixel_nhit])
                inner_nhits.append([inner_nhit])
                outer_nhits.append([outer_nhit])

                # Resolution stuff
                d0_res_vs_pt.append( resolution_dict['d0res_pt'] )
                d0_res_vs_eta.append(resolution_dict['d0res_eta'])
                z0_res_vs_pt.append( resolution_dict['z0res_pt'] )
                z0_res_vs_eta.append(resolution_dict['z0res_eta'])
                pt_res_vs_pt.append( resolution_dict['ptres_pt'] )
                pt_res_vs_eta.append(resolution_dict['ptres_eta'])
                pt_res.append(       resolution_dict['ptres']    )

                # Matched track stuff
                track_pt.append(matched_track_dict['pt'])
                track_eta.append(matched_track_dict['eta'])
                track_phi.append(matched_track_dict['phi'])
                track_theta.append(matched_track_dict['theta'])
                track_d0.append(matched_track_dict['d0'])
                track_z0.append(matched_track_dict['z0'])
                track_ndf.append(matched_track_dict['ndf'])
                track_chi2.append(matched_track_dict['chi2'])

                # Track-matched muon stuff
                pt_match.append(   matched_muon_dict['pt'])
                eta_match.append(  matched_muon_dict['eta'])
                theta_match.append(matched_muon_dict['theta'])
                phi_match.append(  matched_muon_dict['phi'])

            pfo_pt.append(pfo_dict['pt'])
            pfo_eta.append(pfo_dict['eta'])
            pfo_phi.append(pfo_dict['phi'])
            pfo_mu_pt.append(pfo_mu_dict['pt'])
            pfo_mu_eta.append(pfo_mu_dict['eta'])
            pfo_mu_phi.append(pfo_mu_dict['phi'])

            # if there are fake tracks, record them too
            if(has_fake_tracks):
                fake_pt.append(fake_track_dict['pt'])
                fake_eta.append(fake_track_dict['eta'])
                fake_phi.append(fake_track_dict['phi'])
                fake_theta.append(fake_track_dict['theta'])
                fake_d0.append(fake_track_dict['d0'])
                fake_z0.append(fake_track_dict['z0'])
                fake_ndf.append(fake_track_dict['ndf'])
                fake_chi2.append(fake_track_dict['chi2'])
                fake_nhits.append(fake_track_dict['nhits'])
            #     # print(fake_pt)
            event_counter += 1
        reader.close()

    # ############## MANIPULATE, PRETTIFY, AND SAVE HISTOGRAMS #############################
    print("\nSummary statistics:")
    print("Ran over {} events.".format(event_counter))
    # print("Found:")
    # print("\t{} MCPs".format(hists["mcp_pt"].GetEntries()))
    # print("\t{} mu MCPs".format(hists["mcp_mu_pt"].GetEntries()))
    # # print("\tSanity check mcp_mu_pt:", len(mcp_mu_pt))
    # # print("\t%i PFOs"%hists["pfo_pt"].GetEntries())
    # # print("\t%i mu PFOs"%hists["pfo_mu_pt"].GetEntries())
    # print('\t{} matched muon tracks'.format((num_matched_tracks)))
    # print('\t{} duplicates eliminated'.format(num_dupes))
    # # print('\t%i hard radiations discarded'%hard_rad_discard)
    # print('\t{} fake tracks'.format(num_fake_tracks))
    # # print('\t%i GeV'%np.max(mcp_mu_pt))
    return

    # Make a list of all the data you want to save
    data_list = {}
    data_list["mcp_pt"] = mcp_pt
    data_list["mcp_eta"] = mcp_eta
    data_list["mcp_phi"] = mcp_phi
    data_list["mcp_mu_pt"] = mcp_mu_pt
    data_list["mcp_mu_eta"] = mcp_mu_eta
    data_list["mcp_mu_phi"] = mcp_mu_phi

    # data_list["pfo_pt"] = pfo_pt
    # data_list["pfo_eta"] = pfo_eta
    # data_list["pfo_phi"] = pfo_phi
    # data_list["pfo_mu_pt"] = pfo_mu_pt
    # data_list["pfo_mu_eta"] = pfo_mu_eta
    # data_list["pfo_mu_phi"] = pfo_mu_phi

    # data_list["mcp_mu_match_pt"] = mcp_mu_match_pt
    # data_list["mcp_mu_match_eta"] = mcp_mu_match_eta
    # data_list["mcp_mu_match_phi"] = mcp_mu_match_phi
    # data_list["d_mu_dpt"] = d_mu_dpt
    # data_list["d_mu_drelpt"] = d_mu_drelpt
    # data_list["d_mu_dphi"] = d_mu_dphi
    # data_list["d_mu_deta"] = d_mu_deta

    data_list["d0_res"] = d0_res
    data_list["z0_res"] = z0_res
    data_list["nhits"] = nhits
    data_list["pixel_nhits"] = pixel_nhits
    data_list["inner_nhits"] = inner_nhits
    data_list["outer_nhits"] = outer_nhits
    data_list["pt_res_hits"] = pt_res_hits
    data_list["d0_res_vs_pt"] = d0_res_vs_pt
    data_list["d0_res_vs_eta"] = d0_res_vs_eta
    data_list["z0_res_vs_pt"] = z0_res_vs_pt
    data_list["z0_res_vs_eta"] = z0_res_vs_eta
    data_list["pt_res_vs_eta"] = pt_res_vs_eta
    data_list["pt_res_vs_pt"] = pt_res_vs_pt
    data_list["pt_res"] = pt_res
    data_list["pt_match"] = pt_match
    data_list["track_pt"] = track_pt
    data_list["track_eta"] = track_eta
    data_list["eta_match"] = eta_match
    data_list["theta_match"] = theta_match
    data_list["phi_match"] = phi_match
    data_list["track_ndf"] = track_ndf
    data_list["track_chi2"] = track_chi2
    data_list["track_d0"] = track_d0
    data_list["track_z0"] = track_z0

    data_list["LC_pt_match"] = LC_pt_match
    data_list["LC_track_pt"] = LC_track_pt
    data_list["LC_track_eta"] = LC_track_eta
    data_list["LC_eta_match"] = LC_eta_match
    data_list["LC_track_theta"] = LC_track_theta
    data_list["LC_phi_match"] = LC_phi_match
    data_list["LC_ndf"] = LC_ndf
    data_list["LC_chi2"] = LC_chi2
    data_list["LC_d0"] = LC_d0
    data_list["LC_z0"] = LC_z0
    data_list["LC_nhits"] = LC_nhits
    data_list["LC_pixel_nhits"] = LC_pixel_nhits
    data_list["LC_inner_nhits"] = LC_inner_nhits
    data_list["LC_outer_nhits"] = LC_outer_nhits
    data_list["LC_pt_res"] = LC_pt_res
    data_list["LC_dr"] = LC_dr

    data_list["fake_pt"] = fake_pt
    data_list["fake_theta"] = fake_theta
    data_list["fake_eta"] = fake_eta
    data_list["fake_phi"] = fake_phi
    data_list["fake_d0"] = fake_d0
    data_list["fake_z0"] = fake_z0
    data_list["fake_ndf"] = fake_ndf
    data_list["fake_chi2"] = fake_chi2
    data_list["fake_nhits"] = fake_nhits
    data_list["fake_pixel_nhits"] = fake_pixel_nhits
    data_list["fake_inner_nhits"] = fake_inner_nhits
    data_list["fake_outer_nhits"] = fake_outer_nhits

    data_list["h_2d_relpt"] = h2d_relpt

    # After the loop is finished, save the data_list to a .json file
    with open(output_json, 'w') as fp:
        json.dump(data_list, fp)

    # # Draw all the 1D histograms you filled
    # for i, h in enumerate(hists):
    #     c = rt.TCanvas("c%i"%i, "c%i"%i)
    #     hists[h].Draw()
    #     hists[h].GetXaxis().SetTitle(h)
    #     hists[h].GetYaxis().SetTitle("Entries")

    #     # For resolution plots, fit them and get the mean and sigma
    #     if h.startswith("d_mu"):
    #         f = rt.TF1("f%i"%i, "gaus")
    #         f.SetLineColor(rt.kRed)
    #         hists[h].Fit("f%i"%i)
    #         c.SetLogy()
    #         latex = rt.TLatex()
    #         p = f.GetParameters()
    #         latex.DrawLatexNDC(.64, .85, "Mean: %f"%p[1])
    #         latex.DrawLatexNDC(.64, .78, "Sigma: %f"%p[2])
    #     c.SaveAs("plots/%s.png"%h)

    # # Make efficiency plots
    # # In these files, there are at most 1 PFO mu, so matching isn't needed
    # for v in variables:
    #     if v=="n": continue
    #     c = rt.TCanvas("c%s"%v, "c%s"%v)
    #     eff = rt.TEfficiency(hists["mcp_mu_match_"+v], hists["mcp_mu_"+v])
    #     eff.Draw("ape")
    #     rt.gPad.Update()
    #     eff.SetLineWidth(2)
    #     eff.GetPaintedGraph().SetMinimum(0)
    #     eff.GetPaintedGraph().SetMaximum(1)
    #     eff.SetTitle(";%s;Efficiency"%v)
    #     c.SaveAs("plots/eff_%s.png"%v)

    # # Make 2D plot and a TProfile to understand pT resolution v pT
    # c = rt.TCanvas("crelpt2d", "crelpt2d")
    # h_2d_relpt.Draw("colz")
    # h_2d_relpt.GetXaxis().SetTitle("pt")
    # h_2d_relpt.GetYaxis().SetTitle("drelpt")
    # c.SaveAs("plots/d_mu_relpt_2d.png")

    # c = rt.TCanvas("crelpt2dprof", "crelpt2dprof")
    # h_prof = h_2d_relpt.ProfileX("_pfx", 1, -1, "s")
    # h_prof.GetXaxis().SetTitle("pt")
    # h_prof.GetYaxis().SetTitle("drelpt")
    # h_prof.Draw()
    # c.SaveAs("plots/d_mu_relpt_prof.png")

    # for var in hists_2d:
    #     c = rt.TCanvas("c_" + var, "c_" + var)
    #     hists_2d[var].Draw("colz")
    #     hists_2d[var].GetXaxis().SetTitle(variables_2d[var]["xlabel"])
    #     hists_2d[var].GetYaxis().SetTitle(variables_2d[var]["ylabel"])
    #     c.SaveAs("plots/" + var + ".png")

if(__name__=='__main__'):
    main(sys.argv)