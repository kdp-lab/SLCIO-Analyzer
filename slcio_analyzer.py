# Author: Jan T. Offermann (UChicago)
# Based on code written by Leo Rozanov (UChicago)

import sys,os
import pyLCIO
import ROOT as rt
import argparse as ap

import utils.qol_utils.progress_bar as pb
from utils.writers import writers
from utils.track import Track
from utils.misc import GetFourVector, GetNumEventsTotal, ParseInputFiles
from utils.condor.condor import CondorRunner

def check_hard_radiation(mcp, fractional_threshold):
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

class Processor():
    """
    This class basically just provides the "main" function of
    this script -- I've made it like a class simply because
    I find this a convenient way to organize the code, when
    including optional condor functionality.
    """

    def __init__(self, fnames, max_events, verbose, output_filename, mode):
        self.min_dr = 0.005
        self.Bfield = 5. # T, 3.57 for legacy
        self.fractional_threshold = 0. # for checking hardness of radiation from muon
        self.track_pt_min = 0.5 # GeV, I think -Jan
        self.bad_pt_res_threshold = 25

        self.fnames = fnames
        self.max_events = max_events
        self.verbose = verbose
        self.output_filename = output_filename
        self.mode = mode

        self.writer = None

    def SetWriter(self):
        if(self.mode=='json'):
            self.writer = writers.JsonWriter(output_file=self.output_filename)
        else:
            self.writer = writers.RootWriter(output_file=self.output_filename)
        return


    def Run(self):
        self.SetWriter()

        event_counter = 0
        num_matched_tracks = 0
        num_dupes = 0
        num_fake_tracks = 0

        collection_names = [
            "MCParticle",
            # "PandoraPFOs", # NOTE: Not present! Will keep relevant code/comments for now.
            'AllTracks',
            "SiTracks", # Note: Causes crash if AllTracks is not loaded too -- this basically holds pointers to AllTracks collection.
            # "SeedTracks",
            # "SiTracks_Refitted", # NOTE: Not present!
            "MCParticle_SiTracks",
            # "MCParticle_SiTracks_Refitted",
        ]

        track_collection_name = 'SiTracks' # TODO: eventually make this toggleable
        relation_collection_name = 'MCParticle_{}'.format(track_collection_name)
        assert(track_collection_name in collection_names)
        assert(relation_collection_name in collection_names)

        # Treat the hit collections separately -- these might not be present.
        hit_collection_names = [
            "VBTrackerHitsConed",
            "VETrackerHitsConed"
        ]
        hit_collection_mask = {key:True for key in hit_collection_names} # TODO: Not sure if this works as intended? -Jan

        branch_list = collection_names + hit_collection_names
        reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.setReadCollectionNames(branch_list)

        print(self.fnames,self.max_events) # DEBUG
        num_events_total = GetNumEventsTotal(self.fnames,self.max_events)
        print('Looping over {} events.'.format(num_events_total))

        # ############## LOOP OVER EVENTS AND FILL HISTOGRAMS  #############################
        # Loop over events
        for f in self.fnames:
            if self.max_events > 0 and event_counter >= self.max_events: break

            reader.open(f)
            for i,event in enumerate(reader):
                if self.max_events > 0 and event_counter >= self.max_events: break

                # Events are typically quite large, so it is OK to print for each one:
                # this is probably not going to be what slows down the code.
                print('Processing event {}/{}'.format(i+1,num_events_total))

                # Get the collections we care about
                relation_collection = event.getCollection(relation_collection_name)
                relation = pyLCIO.UTIL.LCRelationNavigator(relation_collection)

                mcp_collection = event.getCollection("MCParticle")
                # pfo_collection = event.getCollection("PandoraPFOs")
                track_collection = event.getCollection(track_collection_name)

                hit_collections = []
                for hname in hit_collection_names:
                    if(not hit_collection_mask[hname]):
                        continue
                    try:
                        hit_collections.append(event.getCollection(hname))
                    except:
                        hit_collection_mask[hname] = False
                        if(self.verbose > 0):
                            print('\tDid not find hit collection: {}. Disabling...'.format(hname))
                        pass

                # Make counter variables
                n_mcp_mu = 0
                n_pfo_mu = 0
                # has_pfo_mu = False
                # pfo_mu_vec = 0

                # TODO: No PFO!
                # # Pfos
                # pfo_dict = {
                #     'pt':[],
                #     'eta':[],
                #     'phi':[]
                # }

                # pfo_mu_dict = {
                #     'pt':[],
                #     'eta':[],
                #     'phi':[]
                # }

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

                #TODO: No PFO!
                # # MCP that was matched to PFO.
                # # TODO: Check that this matching is really working,
                # # this dict is filled in a MCP loop if some condition was met
                # # in a previous, separate PFO loop.
                # mcp_mu_match_dict = {
                #     'pt':[],
                #     'eta':[],
                #     'phi':[]
                # }

                # # note this dictionary's structure is not like many of the others
                # d_mu_dict = {
                #     'dpt':[],
                #     'deta':[],
                #     'dphi':[],
                #     'relpt':[],
                #     'pt_relpt':[]
                # }

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

                # LC-matched tracks
                lc_matched_track_dict = {
                    'pt':[],
                    'eta':[],
                    'phi':[],
                    'theta':[],
                    'd0':[],
                    'z0':[],
                    'ndf':[],
                    'chi2':[],
                    'nhits':[],
                    'ptres':[],
                    'pixel_nhit':[],
                    'inner_nhit':[],
                    'outer_nhit':[]
                }

                # LC-matched MCPs
                lc_matched_mcp_dict = {
                    'pt':[],
                    'eta':[],
                    'phi':[]
                }

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
                    'nhits':[],
                    'pixel_nhit':[],
                    'inner_nhit':[],
                    'outer_nhit':[]
                }

                # pfo_mu_index = None
                mcp_muon_index = None
                has_fake_tracks = False
                # ##################################################################
                # # Loop over the reconstructed objects and fill histograms
                # for j,pfo in enumerate(pfo_collection):

                #     FillKinematicDict(pfo,pfo_dict)

                #     # hists["pfo_pt"].Fill(pfo_tlv.Perp())
                #     # hists["pfo_eta"].Fill(pfo_tlv.Eta())
                #     # hists["pfo_phi"].Fill(pfo_tlv.Phi())


                #     if abs(pfo.getType())==13:
                #         # hists["pfo_mu_pt"].Fill(pfo_tlv.Perp())
                #         # hists["pfo_mu_eta"].Fill(pfo_tlv.Eta())
                #         # hists["pfo_mu_phi"].Fill(pfo_tlv.Phi())

                #         FillKinematicDict(pfo,pfo_mu_dict)
                #         # NOTE: This this only keeps the last instance of pfo_mu. Probably OK, since we're using muon gun samples -- but good to keep in mind. -Jan
                #         pfo_mu_index = j # keep track of this - will use in track loop later below.
                #         n_pfo_mu += 1
                #         has_pfo_mu = True
                # ##################################################################

                ##################################################################
                # Loop over the truth objects and fill histograms
                for j,mcp in enumerate(mcp_collection):
                    pb.printProgressBar(
                        j,
                        len(mcp_collection),
                        prefix='\tMCPs',
                        suffix='Complete'
                    )
                    FillKinematicDict(mcp,mcp_dict)

                    if(abs(mcp.getPDG())==13 and mcp.getGeneratorStatus()==1):

                        # Check if the muon radiated significant energy
                        hard_rad = check_hard_radiation(mcp, self.fractional_threshold)

                        FillKinematicDict(mcp,mcp_mu_dict)
                        mcp_muon_index = j
                        n_mcp_mu += 1

                        mcp_vec = GetFourVector(mcp)
                        if(mcp_vec.Pt() > self.track_pt_min): # Remove ultra-low pt MCPs # TODO: Comment originally implied removing low-pT tracks but this is applied to MCPs! -Jan

                            # TODO: Looping over "tracks" sometimes causes issues, elements are then reported as being of parent class LCObject. Could this correspond w/ empty list?
                            tracks = relation.getRelatedToObjects(mcp)
                            try: # to deal with the above-mentioned issue
                                for k,track in enumerate(tracks):

                                    track_container = Track(track,self.Bfield)

                                    track_vec = track_container.GetVector()
                                    dr = rt.Math.VectorUtil.DeltaR(track_vec,mcp_vec)
                                    ptres = (mcp_vec.Pt() - track_vec.Pt()) / mcp_vec.Pt()

                                    FillKinematicDict(track_container,lc_matched_track_dict)
                                    FillKinematicDict(mcp,lc_matched_mcp_dict)

                                    # Fill some extra things
                                    lc_matched_track_dict['ptres'].append(ptres)
                                    lc_matched_track_dict['dr'].append(dr)

                                    if(len(hit_collections) > 0):
                                        LC_pixel_nhit, LC_inner_nhit, LC_outer_nhit = NHitsPerLayer(track,hit_collections[0])
                                        lc_matched_track_dict['pixel_nhit'].append([LC_pixel_nhit])
                                        lc_matched_track_dict['inner_nhit'].append([LC_inner_nhit])
                                        lc_matched_track_dict['outer_nhit'].append([LC_outer_nhit])

                                    num_matched_tracks += 1
                                    if hard_rad: # TODO: Not sure this is correct? Doesn't look like anything is being discarded. -Jan
                                        hard_rad_discard += 1
                            except:
                                pass

                            # For events in which a PFO mu was reconstructed, fill histograms that will
                            # be used for efficiency. Both numerator and denominator must be filled with truth values!
                            # Also fill resolution histograms
                            # if has_pfo_mu:
                            #     # hists["mcp_mu_match_pt"].Fill(mcp_tlv.Perp())
                            #     # hists["mcp_mu_match_eta"].Fill(mcp_tlv.Eta())
                            #     # hists["mcp_mu_match_phi"].Fill(mcp_tlv.Phi())

                            #     FillKinematicDict(mcp, mcp_mu_match_dict)

                            #     pfo_mu_vec = GetFourVector(pfo_collection[pfo_mu_index])
                            #     # hists["d_mu_dpt"].Fill(pfo_mu_vec.Perp() - mcp_tlv.Perp())
                            #     # hists["d_mu_drelpt"].Fill((pfo_mu_vec.Perp() - mcp_tlv.Perp())/mcp_tlv.Perp())
                            #     # hists["d_mu_deta"].Fill(pfo_mu_vec.Eta() - mcp_tlv.Eta())
                            #     # hists["d_mu_dphi"].Fill(pfo_mu_vec.Phi() - mcp_tlv.Phi())
                            #     # h_2d_relpt.Fill(mcp_tlv.Perp(), (pfo_mu_vec.Perp() - mcp_tlv.Perp())/mcp_tlv.Perp())
                            #     d_mu_dict['dpt'].append(pfo_mu_vec.Pt() - mcp_vec.Pt())
                            #     d_mu_dict['drelpt'].append((pfo_mu_vec.Pt() - mcp_vec.Pt())/mcp_vec.Pt())
                            #     d_mu_dict['deta'].append(pfo_mu_vec.Eta() - mcp_vec.Eta())
                            #     d_mu_dict['dphi'].append(pfo_mu_vec.Phi() - mcp_vec.Phi())
                            #     d_mu_dict['pt_relpt'].append([mcp_vec.Pt(), (pfo_mu_vec.Pt() - mcp_vec.Pt())/mcp_vec.Pt()])
                ##################################################################

                pb.printProgressBar(
                    len(mcp_collection),
                    len(mcp_collection),
                    prefix='\tMCPs',
                    suffix='Complete'
                )
                if(n_mcp_mu > 1):
                    print('\tWarning: Found {} truth-level muons in event! Skipping...'.format(n_mcp_mu))

                ##################################################################
                # Loop over the track objects and fill histograms for D0, Z0, and hit counts
                # TODO: I don't entirely understand why some parts of this loop exist. Didn't we already loop over tracks within the loop above? -Jan
                counter = 0
                max_hits = 0
                best_track = None

                track_print_chunk = int(len(track_collection) / 200)
                progress_bar = pb.ProgressBar(
                        prefix='\tTracks',
                        suffix='Complete'
                )

                for j,track in enumerate(track_collection):

                    if(j%track_print_chunk==0):
                        progress_bar.Print(j,len(track_collection))

                    track_container = Track(track,self.Bfield)
                    # Get the deltaR between each track and the truth muon.
                    # TODO: This is notably different than in the old code, which effectively
                    #       checked dR against mcp_vec (which would be whatever was the last
                    #       truth particle from the above loops!). #TODO: FIX THIS!!!
                    dr = rt.Math.VectorUtil.DeltaR(track_container.GetVector(),mcp_vec)

                    # Fake tracks
                    if(len(relation.getRelatedFromObjects(track)) == 0): # If there's no associated truth muon
                        # has_fake_tracks = True
                        FillKinematicDict(track_container,fake_track_dict)

                        fake_pixel_nhit = 0
                        fake_inner_nhit = 0
                        fake_outer_nhit = 0
                        if(len(hit_collections) > 0):
                            fake_pixel_nhit, fake_inner_nhit, fake_outer_nhit = NHitsPerLayer(track,hit_collections[0])
                            fake_track_dict['pixel_nhit'].append(fake_pixel_nhit)
                            fake_track_dict['inner_nhit'].append(fake_inner_nhit)
                            fake_track_dict['outer_nhit'].append(fake_outer_nhit)
                        num_fake_tracks += 1
                    else:
                        pass # TODO: Could do something with LC relations here?

                    # Also find track nearest to truth-level muon in (eta,phi), if it exists!
                    if(mcp_muon_index is not None):
                        muon_vec = GetFourVector(mcp_collection[mcp_muon_index])
                        dr = rt.Math.VectorUtil.DeltaR(track_container.GetVector(), muon_vec)

                        if dr < self.min_dr: # Do dR check first, then do nhits check
                            if track_container.GetNHits() > max_hits:
                                max_hits = track_container.GetNHits()
                                best_track = track_container
                            counter += 1
                            if counter > 1:
                                num_dupes += 1
                            # print("More than one track in event! # of dupes:", num_dupes)
                ##################################################################################

                # Compute some resolution stuff, using the track most closely dR-matched to the truth-level muon
                if best_track is not None:

                    muon_vec = GetFourVector(mcp_collection[mcp_muon_index])
                    ptres = (muon_vec.Pt() - best_track.GetVector().Pt()) / muon_vec.Pt()

                    FillKinematicDict(best_track,matched_track_dict)
                    FillKinematicDict(mcp_collection[mcp_muon_index],matched_muon_dict)
                    FillResolutionDict(mcp_collection[mcp_muon_index], best_track, resolution_dict)

                    # pixel_nhit, inner_nhit, outer_nhit = NHitsPerLayer(track,hit_collections[0])

                ##################################################################
                progress_bar.Print(len(track_collection),len(track_collection))

                if n_pfo_mu > 1:
                    print('\tWarning: Found {} reconstructed muons.'.format(n_pfo_mu))

                # Now fill a bunch of things.
                print('\t\tFilling.')

                for key,val in mcp_dict.items():
                    self.writer.Append('mcp_{}'.format(key),val)

                for key,val in mcp_mu_dict.items():
                    self.writer.Append('mcp_mu_{}'.format(key),val)

                for key,val in matched_track_dict.items():
                    self.writer.Append('dr_matched_track_{}'.format(key),val)

                for key,val in matched_muon_dict.items():
                    self.writer.Append('dr_matched_muon_{}'.format(key),val)

                for key,val in resolution_dict.items():
                    self.writer.Append('dr_matched_resolution_{}'.format(key),val)

                for key,val in lc_matched_track_dict.items():
                    self.writer.Append('lc_matched_track_{}'.format(key),val)

                for key,val in lc_matched_mcp_dict.items():
                    self.writer.Append('lc_matched_mcp_{}'.format(key),val)

                for key,val in fake_track_dict.items():
                    self.writer.Append('fake_track_{}'.format(key),val)

                if(self.mode != 'json'):
                    self.writer.FlushBuffersToTree()

                event_counter += 1
            reader.close()
        self.writer.Write()

def main(args):
    rt.gROOT.SetBatch()

    parser = ap.ArgumentParser()
    parser.add_argument('-i','--inputFile',type=str,help='Input text file listing input files, or a glob-compatible string.',required=True)
    parser.add_argument('-n','--nEvents',type=int,default=-1)
    parser.add_argument('-v','--verbose',type=int,default=0)
    parser.add_argument('-o','--outputFile',type=str,default='slcio_analyser.json')
    parser.add_argument('-m','--mode',type=str,default='JSON')
    parser.add_argument('-c','--condor',type=int,default=0, help='If >0, creates jobs for condor instead of running locally.')

    # Some condor-specific args
    parser.add_argument('-r','--runDir',type=str,default='run', help='Run directory for jobs. [condor only]')
    parser.add_argument('-O','--outputDir',type=str,default=None, help='Output directory for jobs. [condor only]')

    args = vars(parser.parse_args())

    fnames = ParseInputFiles(args['inputFile'])
    max_events = args['nEvents'] # Set to -1 to run over all events
    verbose = args['verbose'] # integer
    output_filename = args['outputFile']
    mode = args['mode'].lower()
    use_condor = args['condor'] > 0

    run_dir = args['runDir']
    output_dir = args['outputDir']

    if(use_condor):

        assert(output_dir is not None)

        # Creating condor jobs.
        condor_runner = CondorRunner()

        this_dir = os.path.dirname(os.path.realpath(__file__))
        template = '{}/utils/condor/template.sub'.format(this_dir)
        executable = '{}/utils/condor/condor_job.sh'.format(this_dir)
        # payload_contents = [
        #     '{}/slcio_analyzer.py'.format(this_dir),
        #     '{}/utils'.format(this_dir)
        # ]
        payload_contents = [
            'slcio_analyzer.py',
            'utils'
        ]
        arguments_file = 'arguments.txt'

        condor_runner.SetScriptDirectory(this_dir) # useful specifically for things like the tar command it runs
        condor_runner.SetRunDirectory(run_dir)
        condor_runner.SetOutputDirectory(output_dir)
        condor_runner.SetOutputName(output_filename)
        condor_runner.SetBatchName('SLCIO-Analyzer')
        condor_runner.SetMode('OSG')
        condor_runner.run(template,executable,payload_contents,args['inputFile'],arguments_file)

    else:
        # Running locally.
        processor = Processor(fnames,max_events,verbose,output_filename,mode)
        processor.Run()
    return

if(__name__=='__main__'):
    main(sys.argv)