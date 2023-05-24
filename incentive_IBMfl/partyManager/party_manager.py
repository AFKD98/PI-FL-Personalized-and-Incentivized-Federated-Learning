import os
import sys
import subprocess
import time
import gc

#initialize main method

def kill_all(parties):
    for party in parties:
        party.kill()
    parties = []

def store_output(parties, output_path):
    for party in parties:
        party.wait()
        with open(output_path, 'a') as f:
            f.write(party.stdout.read().decode('utf-8'))
            
def store_output_in_different_files(parties, output_paths):
    party_no = 0
    for party in parties:
        party.wait()
        with open(output_paths[party_no], 'a') as f:
            f.write(party.stdout.read().decode('utf-8'))
        party_no += 1

def print_output(parties):
    for party in parties:
        print(party)
        # break
      
def main():
    #launch subprocesses
    gc.collect()

    parties = []
    output_paths = []
    start = int(sys.argv[1])
    no_of_parties = 25 # for testing
    # subprocess.run(f'conda activate ibm_incentive'.split(),
    # shell=True, executable='/bin/bash', check=True)
    
    for party_no in range(start, start + no_of_parties):
        parties.append(subprocess.Popen('bash -c conda activate ibm_incentive; python -m ibmfl.party.party examples/configs/fedavg/pytorch/config_party{}.yml'.format(party_no), shell=True))
        # capture_output = True, text = True, shell=True))
        output_paths.append('party_logs/party{}.txt'.format(party_no))

    #wait for all subprocesses to finish
    #print_output(parties)
    
    for party in parties:
        try:
            party.wait()
        except KeyboardInterrupt:
            try:
                party.kill()
            except OSError:
                pass
            party.wait()
    print_output(parties)
    #get outputs from subprocesses
    # store_output_in_different_files(parties, output_paths)
    #kill subprocesses
    # kill_all(parties)
    #clear cache
    gc.collect()
    
    
if __name__ == '__main__':
    main()