import os
import subprocess as sub
import numpy as np

class CondorRunner:
    """
    Class based on some ATLAS work I did
    for the 2024 DV+MET analysis. -Jan T. Offermann
    """
    def __init__(self):
        self.run_dir = 'run'
        self.ncpu = 1
        self.memory = 4096 # MiB
        self.out_dir = None
        self.SetScriptDirectory(os.path.dirname(os.path.realpath(__file__)) + '/../../')
        self.SetNFilesPerJob(1)
        self.SetShortQueue(False)

    def SetRunDirectory(self,val):
        self.run_dir = val

    def SetScriptDirectory(self,val):
        self.script_directory = val

    def SetInputListFile(self,val):
        self.input_list_file = val

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
        if(require_cvmfs): requirements.append("HAS_CVMFS =?= TRUE")
        requirements += ["machine != \"{}\"".format(x) for x in blacklist]
        requirements = "(" + " && ".join(requirements) + ")"
        return requirements

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
                new_line = new_line.replace('$SHORT_QUEUE',short_queue)
                f.write(new_line)
        return

    def _write_arguments_file(self,arguments_file,njobs,input_list_filename):
        with open('{}/{}'.format(self.run_dir,arguments_file),'w') as f:
            for i in range(njobs):
                output_name = '{}_job{}'.format(self.out_name,str(i).zfill(3))

                # prepare the argument string.
                arg_string = '{i}'
                arg_string = arg_string.format(
                    i=input_list_filename
                )
                f.write(arg_string + '\n')
        return

    def run(self,condor_template, condor_executable, payload_contents, input_list_filename='inputs.txt',arguments_file='arguments.txt'):
        """
        Function for preparing condor jobs.
        """

        self.input_list_file = input_list_filename # TODO: clean up

        # Create the output directory.
        if(self.out_dir is None):
            raise ValueError("CondorRunner: Output directory is not set.")
        os.makedirs(self.out_dir,exist_ok=True)

        # Get directory that this script is sitting inside.
        # this_dir = os.path.dirname(os.path.abspath(__file__))

        # Gather the necessary files that will be packaged up and sent
        # to the job. NOTE: It is up to the job to unpack these!
        payload = 'payload.tar.gz'
        command = ['tar','-czf',payload] + ['-C',self.script_directory] + payload_contents
        print('Running command: ',' '.join(command))
        sub.check_call(command)

        # make the directory from which the condor jobs will be run
        # self.run_dir = 'run'
        os.makedirs(self.run_dir)

        # move payload to the run directory
        command = ['mv',payload,self.run_dir]
        sub.check_call(command)

        # copy the original input_list_file into the run directory -- it will be split up for the jobs
        command = ['cp',self.input_list_file,self.run_dir]
        sub.check_call(command)

        # parse the input_list_file, determine number of jobs and files for each job
        with open(self.input_list_file,'r') as f:
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
            input_list_file_mini = '{}/{}/{}'.format(self.run_dir,job_dir,input_list_filename)
            with open(input_list_file_mini,'w') as f:
                for entry in files:
                    f.write(entry)

        # Make a file containing the arguments for the jobs. I find this easier to do than writing things in
        # the condor submission file, since we will keep all the arguments in one place for easy access.
        # Each job's output will have a unique name, since the condor jobs will send them all to the same output directory
        # and we want to avoid naming collisions.
        self._write_arguments_file(arguments_file,njobs,input_list_filename)

        # Fetch the condor submission template, and fill it in appropriately.
        self._create_submision_file(condor_template)

        # Fetch the condor executable.
        command = ['cp',condor_executable,self.run_dir]
        sub.check_call(command)
        print('Condor jobs ready for submission from {}.'.format(self.run_dir))
        return
