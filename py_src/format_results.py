import os
import glob
import json
import re

'''
This script assumes that the results are stored in the `logs` folder in the following format:
    <action description>: <floating point value time>
In the end, produces the results in the `logs/summarized.json` as follows:
    <action description>: [<average time>, <variance of times>]
It follows, that the <action description> needs to be descriptive enough and that it needs
to be unique in the whole file.
It also produces a `logs/summarized_final.json` file with only the average total execution
times of all the algorithms (it detects these as they contain the `TOTAL` word inside the 
action description)
'''

def compute_statistics(lst):
    '''
    Computes mean and variance and returns them as a list
    '''
    mean = sum(lst) / len(lst)
    var = sum((x - mean) ** 2 for x in lst) / len(lst)
    return [mean, var]

def format_results():
    '''
    Reads all the files in the `logs` folder and creates the two files described above
    '''
    total_times_dict = {}
    files_count = 0

    for file in glob.glob('logs/*.txt'):
        files_count += 1
        print(f'Processing file {file}')
        f = open(file, 'r')
        for line in f.readlines():
            if re.search('.*\d+\.\d+\n?$', line) is None: # check if line ends with float
                continue
            tmp = line.split()
            action_description = ' '.join(tmp[:-1])
            time = float(tmp[-1])
            if action_description not in total_times_dict:
                total_times_dict[action_description] = []
            total_times_dict[action_description].append(time)
        f.close()
    assert all(len(v) == files_count for v in total_times_dict.values())
    total_times_dict = {k: compute_statistics(v) for k, v in total_times_dict.items()}
    
    # dump the results to file
    with open(os.path.join('logs', 'summarized.json'), 'w') as f:
        json.dump(total_times_dict, f, indent=2)
    with open(os.path.join('logs', 'summarized_final.json'), 'w') as f:
        # omit the variance, as it typically is in the order of 1e-9
        json.dump({k: round(v[0], 3) for k, v in total_times_dict.items() if 'TOTAL' in k}, f, indent=2)


if __name__ == '__main__':
    format_results()