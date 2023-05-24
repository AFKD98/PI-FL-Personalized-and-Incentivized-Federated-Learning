import os
import sys
import subprocess
import time
import gc
from multiprocessing.pool import ThreadPool
from ibmfl.party.party import Party
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

def execute_async(config_path):
        p = Party(config_file=config_path)
        p.start()
        p.register_party()
        return p
    
def main():
    #create parties
    pool = ThreadPool(processes=50)
    parties = []
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    for party_no in range(start, end):
        config_path = 'examples/configs/fedavg/pytorch/config_party{}.yml'.format(party_no)
        parties.append(pool.apply_async(execute_async, (config_path,)))
        # parties.append(subprocess.Popen('bash -c conda activate ibm_incentive; python -m ibmfl.party.party examples/configs/fedavg/pytorch/config_party{}.yml'.format(party_no), shell=True))
        # capture_output = True, text = True, shell=True))
    try:
        while(True):
            cmd = input("Enter command: ")
            for p in parties:
                p.send_signal(cmd)
    except KeyboardInterrupt:
        pool.close()
        pool.join()
        results = [r.get() for r in parties]
        print(results)


# def main():
#     #launch subprocesses
#     gc.collect()

#     parties = []
#     output_paths = []
#     start = int(sys.argv[1])
#     no_of_parties = 25 # for testing
#     # subprocess.run(f'conda activate ibm_incentive'.split(),
#     # shell=True, executable='/bin/bash', check=True)
#     pool = ThreadPool(processes=30)
#     for party_no in range(start, 50):
#         parties.append(pool.apply_async(execute_async, (party_no,)))
#         # parties.append(subprocess.Popen('bash -c conda activate ibm_incentive; python -m ibmfl.party.party examples/configs/fedavg/pytorch/config_party{}.yml'.format(party_no), shell=True))
#         # capture_output = True, text = True, shell=True))
#         output_paths.append('party_logs/party{}.txt'.format(party_no))
#     pool.close()
#     pool.join()
#     results = [r.get() for r in parties]
#     print(results)
#     #wait for all subprocesses to finish
#     #print_output(parties)
    
#     for party in parties:
#         try:
#             party.wait()
#         except KeyboardInterrupt:
#             try:
#                 party.kill()
#             except OSError:
#                 pass
#             party.wait()
#     print_output(parties)
#     #get outputs from subprocesses
#     # store_output_in_different_files(parties, output_paths)
#     #kill subprocesses
#     # kill_all(parties)
#     #clear cache
#     gc.collect()
    
    
if __name__ == '__main__':
    main()