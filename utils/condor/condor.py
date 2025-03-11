import os
import subprocess as sub
import numpy as np

class CondorRunner:
    """
    Class based on some ATLAS work I did
    for the 2024 DV+MET analysis. -Jan T. Offermann
    """
    def __init__(self,style='arguments'):
        self.run_dir = 'run'
        self.ncpu = 1
        self.memory = 4096 # MiB
        self.out_dir = None
        self.SetScriptDirectory(os.path.dirname(os.path.realpath(__file__)) + '/../../')
        self.SetNFilesPerJob(1)
        self.SetShortQueue(False)
        self.mode='UCAF'
        self.inputs = []
        self.payload = '../payload.tar.gz' # name for tar containing input script and libraries - by default it sits one directory up from the initialdir
        self.SetStyle(style)

        # stuff related to "queue" mode
        self.queue_vars = None
        self.queue_lists = None
        self.queue_string = None

    def SetPayload(self,val):
        self.payload = val

    def SetInputs(self,val,append_payload=True):
        self.inputs = val
        if(append_payload):
            self.inputs += [self.payload]

    def SetMode(self,val):
        self.mode = val

    def SetQueueVars(self,val):
        self.queue_vars = val

    def SetQueueLists(self,val):
        self.queue_lists = val

    def SetRunDirectory(self,val):
        self.run_dir = val



    def SetScriptDirectory(self,val):
        self.script_directory = val

    def SetInputListFile(self,val):
        self.input_data_list_file = val

    def SetNFilesPerJob(self,val):
        self.nfiles_per_job = val

    def SetOutputName(self,val):
        self.out_name = val

    def SetOutputDirectory(self,val):
        self.out_dir = val

    def SetSignalRegion(self,val):
        self.sig_region = val

    def SetNCPU(self,val):
        self.ncpu = val

    def SetMemory(self,val):
        self.memory = val

    def SetBatchName(self,val):
        self.batch_name = val

    def SetShortQueue(self,val):
        self.short_queue = val

    def _short_queue(self):
        """
        Something quite specific to the
        UChicago Analysis Facility,
        flags condor jobs to use the dedicated
        short queue where they must run for no
        longer than 3 hours.
        """
        if(self.short_queue):
            print('\nUsing short queue.\n')
            return '    +queue="short"'
        else:
            return ''

    def _fetch_requirements(self,require_cvmfs=True,blacklist_file=None):
        """
        By default, requires condor worker to have CVMFS access.
        (this is at least based on UChicago Analysis Facility
        condor worker setup -- not sure how this works elsewhere).
        Can also optionally blacklist certain condor workers,
        which can be useful to deal with machine-specific issues.
        """
        blacklist = []
        if(blacklist_file is not None):
            with open(blacklist_file,'r') as f:
                blacklist= blacklist_file.readlines()
        blacklist = [x.strip().strip('\n') for x in blacklist]
        requirements = []
        if(require_cvmfs):
            if(self.mode=='UCAF'):
                requirements.append("HAS_CVMFS =?= TRUE")
            elif(self.mode=='OSG'):
                requirements.append('HAS_CVMFS_unpacked_cern_ch') #'(HAS_SINGULARITY ) && ( HAS_CVMFS_unpacked_cern_ch'
        requirements += ["machine != \"{}\"".format(x) for x in blacklist]
        requirements = "(" + " && ".join(requirements) + ")"
        return requirements

    def SetStyle(self,style):
        """
        This function determines how the condor job handles arguments:
        are they provided in an explicit arguments file from which condor
        queues jobs, or are they explicitly listed within the condor submission
        script? The latter is a bit clunky but allows for some advanced behaviour,
        such as queueing command-line arguments as well as file inputs for the
        condor transfer protocol (which is useful if the workers cannot access
        the filesystem where the input data lives).
        """
        self.style = style
        if(self.style == 'queue'):
            assert(self.nfiles_per_job == 1) # will break otherwise

    def _create_submision_file(self,template):

        with open(template,'r') as f:
            lines = f.readlines()

        submission_file = '{}/condor.sub'.format(self.run_dir)
        short_queue = self._short_queue()
        with open(submission_file,'w') as f:
            for line in lines:
                new_line = line
                new_line = new_line.replace('$BATCH_NAME',self.batch_name)
                new_line = new_line.replace('$REQUIREMENTS',self._fetch_requirements())
                new_line = new_line.replace('$NCPU',str(self.ncpu))
                new_line = new_line.replace('$MEM',str(self.memory))
                new_line = new_line.replace('$INPUTS',', '.join(self.inputs))

                # The short queue is something UCAF-specific
                new_line = new_line.replace('$SHORT_QUEUE',short_queue)

                if(self.style != 'arguments'):
                    new_line = new_line.replace('$QUEUE',self.queue_string)
                f.write(new_line)
        return

    def _write_arguments_file(self,arguments_file,njobs,input_list_filename, **kwargs):
        """
        This function will need to be customized if using this CondorRunner elsewhere,
        this is where we actually write the file containing job arguments.
        """
        with open('{}/{}'.format(self.run_dir,arguments_file),'w') as f:
            for i in range(njobs):
                file_extension = self.out_name.split('.')[-1] # TODO: Fragile code?
                filename_no_extension = self.out_name.split('.')[0]
                output_name = '{}_job{}.{}'.format(filename_no_extension,str(i).zfill(3),file_extension)
                output_path = '{}/{}'.format(kwargs['output_directory'],output_name)
                # prepare the argument string.
                arg_string = '{i} {O}'
                arg_string = arg_string.format(
                    i=input_list_filename,
                    O=output_path
                )
                f.write(arg_string + '\n')
        return

    def _create_queue_string(self,vars,lists):
        queue_string = 'queue ' + ', '.join(vars) + ' from (\n'
        # a bit hacky, assuming lists is a list of lists.
        # e.g. [[1,2],['a','b']] where we want values then queued as
        # 1 a
        # 2 b
        #
        l = len(lists[0])
        for i in range(l):
            queue_string +=  ', '.join([x[i] for x in lists]) + '\n'
        queue_string += ')'
        self.queue_string = queue_string

    def run(self,condor_template, condor_executable, payload_contents,**kwargs):
        """
        Function for preparing condor jobs.
        """

        # Create the output directory.
        if(self.style=='arguments'):
            if(self.out_dir is None):
                raise ValueError("CondorRunner: Using \'arguments\' job creation method, but output directory is not set.")
            os.makedirs(self.out_dir,exist_ok=True)

        # Get directory that this script is sitting inside.
        # this_dir = os.path.dirname(os.path.abspath(__file__))

        # make the directory from which the condor jobs will be run
        os.makedirs(self.run_dir)

        # Gather the necessary files that will be packaged up and sent
        # to the job. NOTE: It is up to the job to unpack these!
        command = ['tar','-czf',self.payload] + ['-C',self.script_directory] + payload_contents
        sub.check_call(command)

        # move self.payload to the run directory
        try:
            command = ['mv',self.payload,self.run_dir]
            sub.check_call(command)
        except:
            pass

        # copy the original input_data_list_file into the run directory -- it will be split up for the jobs
        command = ['cp',self.input_data_list_file,self.run_dir]
        sub.check_call(command)

        # parse the input_data_list_file, determine number of jobs and files for each job
        with open(self.input_data_list_file,'r') as f:
            input_files = f.readlines()
        nfiles = len(input_files)

        njobs = np.divmod(nfiles,self.nfiles_per_job)
        if(njobs[1] == 0): njobs = njobs[0]
        else: njobs = njobs[0] + 1

        for i in range(njobs):
            if(i != njobs - 1): files = input_files[i * self.nfiles_per_job : (i+1) * self.nfiles_per_job]
            else: files = input_files[i * self.nfiles_per_job:]

            job_dir = 'job{}'.format(i)
            os.makedirs('{}/{}'.format(self.run_dir,job_dir))
            input_list_file_mini = '{}/{}/{}'.format(self.run_dir,job_dir,self.input_data_list_file.split('/')[-1])
            with open(input_list_file_mini,'w') as f:
                for entry in files:
                    f.write(entry)

        if(self.style == 'arguments'):
            # Make a file containing the arguments for the jobs. I find this easier to do than writing things in
            # the condor submission file, since we will keep all the arguments in one place for easy access.
            # Each job's output will have a unique name, since the condor jobs will send them all to the same output directory
            # and we want to avoid naming collisions.

            self._write_arguments_file(kwargs['arguments_file'],njobs,self.input_data_list_file,output_directory=self.out_dir)
            # self.SetInputs([self.input_data_list_file]) # self.payload is one directory up, since initialdir will be the individual job dirs

        else:
            # create the queue string, that will be written into the submission file
            # self._create_queue_lists()
            self._create_queue_string(self.queue_vars,self.queue_lists)

        # Fetch the condor submission template, and fill it in appropriately.
        self._create_submision_file(condor_template)

        # Fetch the condor executable.
        command = ['cp',condor_executable,self.run_dir]
        sub.check_call(command)
        print('Condor jobs ready for submission from {}.'.format(self.run_dir))
        return
