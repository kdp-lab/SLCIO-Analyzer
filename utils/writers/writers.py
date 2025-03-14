import json
import ROOT as rt
import numpy as np

class JsonWriter():
    """
    A simple class for writing to JSON files.
    These are plaintext and will be quite large,
    plus all the data is stored in memory and
    only flushed at the end.
    A different way of writing data -- likely
    using ROOT -- would be much preferrable.
    """
    def __init__(self,output_file='output.json'):
        self.output_filename = output_file
        self.data_dict = {}

    def CreateBuffer(self,key):
        self.data_dict[key] = []

    def Append(self,key,val):
        if(key not in self.data_dict.keys()):
            self.CreateBuffer(key)
        self.data_dict[key].append(val)

    def Write(self):
        with open(self.output_filename, 'w') as fp:
            json.dump(self.data_dict, fp, indent=4)

class RootWriter():
    """
    A simple class for writing to ROOT files,
    specifically dumping information into a TTree.
    Compared to the JsonWriter, this is a little more
    involved since there's a bit more overhead in
    setting up the ROOT file. However, we will benefit
    from its smaller size (binary vs. plaintext), as well
    as the fact that the data will be continually flushed
    to the file instead of all sitting in memory before
    the Write() function is called.
    """

    def __init__(self,output_file='output.root',tree_name='ntuple'):
        self.output_name = output_file
        self.tree_name = tree_name
        self.data_dict = {}

        self.root_file = rt.TFile(output_file,'RECREATE') # will overwrite output file if it exists
        self.tree = rt.TTree(self.tree_name,self.tree_name)

    def CreateBuffer(self,key,val,datatype=None):
        if(type(val) == int):
            self.data_dict[key] = np.zeros(1,dtype=int)
            self.tree.Branch(key,self.data_dict[key],'{}/I'.format(key))
        elif(type(val) == float):
            self.data_dict[key] = np.zeros(1,dtype=float)
            self.tree.Branch(key,self.data_dict[key],'{}/D'.format(key))
        elif(type(val) in (list,np.ndarray)): # assume a vector -- will be a bit hacky, admittedly!
            if(len(val) == 0): # TODO: Dangerous case -- don't want to skip since this'll cause issues if this *never* gets filled, and we hadd with another tree where it was...
                if(datatype is None): # will allow user to explicitly state the data type
                # NOTE: will try to infer the data type based on the key name. This is pretty hacky!
                    if(('pt' in key) or ('eta' in key) or ('phi' in key) or ('theta' in key) or ('d0' in key) or ('z0' in key) or ('res' in key) or ('chi2' in key)):
                        datatype = 'double'
                    elif(('nhit' in key) or ('ndf' in key)):
                        datatype='int'

                if(datatype is not None):
                    self.data_dict[key] = rt.std.vector(datatype)()
                    self.tree.Branch(key,self.data_dict[key])
                else:
                    raise ValueError("Error in RootWriter.CreateBuffer: Created with empty list, but type cannot be inferred. Key = {}".format(key))

            elif(type(val[0]) == int):
                self.data_dict[key] = rt.std.vector('int')()
                self.tree.Branch(key,self.data_dict[key])
            elif(type(val[0]) == float):
                self.data_dict[key] = rt.std.vector('double')() # NOTE: double vs float? Need to be consistent throughout
                self.tree.Branch(key,self.data_dict[key])
            elif(type(val[0]) == list): # getting a bit complicated!
                if(len(val[0]) == 2 and type(val[0][0] == float) and type(val[0][1] == float)):
                    self.data_dict[key] = rt.std.vector(rt.std.vector('double'))()
                    self.tree.Branch(key,self.data_dict[key])
                else:
                    raise ValueError('Error in RootWriter.CreateBuffer: Identified 2D list/array, but unable to identify its type. Key = {}'.format(key))
            else:
                raise ValueError('Error in RootWriter.CreateBuffer: Identified list/array, but unable to identify its type. Key = {}'.format(key))
        else:
            raise ValueError('Error in RootWriter.CreateBuffer: Unable to identify buffer type. Key = {}'.format(key))
        return

    def WriteToBuffer(self,key,val):
        if(key not in self.data_dict.keys()):
            self.CreateBuffer(key,val)

        if(type(val) in (int,float)):
            self.data_dict[key][0] = val

        else: # assuming rt.std.vector
            for entry in val:
                self.data_dict[key].push_back(entry)
        return

    def Append(self,key,val):
        self.WriteToBuffer(key,val)
        return

    def FlushBuffersToTree(self):
        self.tree.Fill()

        # Clear out all the vector-type branches, as they are filled via push_backs
        for key,val in self.data_dict.items():
            try: # lazy way to do this
                val.clear()
            except:
                pass
        return

    def Write(self):
        self.root_file.cd() # just in case (not sure if needed?)
        self.tree.Write()
        self.root_file.Close()
