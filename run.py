import numpy as np
from ocsort import OCSORTTracker
from evaluate import evaluate
from utils import count_time
import configparser
import os


# DATASET = 'MOT17'
DATASET = 'DanceTrack'
SPLIT = 'test'
SEQS = ['dancetrack0003']
# SEQS = None
DETECTION_FOLDER = 'ocsort_x_dance'

@count_time
def run(seq):
    print(seq)
    tracker = OCSORTTracker()
    os.makedirs('outputs/ocsort-self', exist_ok=True)
    file = open(f'outputs/ocsort-self/{seq}.txt', 'w')
    detections = np.loadtxt(f'detections/{DETECTION_FOLDER}/{seq}.txt', delimiter=',')
    config = configparser.ConfigParser()
    config.read(f'../../.Datasets/{DATASET}/{SPLIT}/{seq}/seqinfo.ini')
    
    for frame_number in range(1, int(config['Sequence']['seqLength']) + 1):
        dets = detections[detections[:, 0] == frame_number][:, 1:]
        tracker.update(dets)
        for output in tracker.get_outputs():
            file.write(f'{output}\n')
    file.close()


if __name__ == '__main__':
    if SEQS:
        seqs = SEQS
    else:
        seqs = os.listdir(f'../../.Datasets/{DATASET}/{SPLIT}/')
    seqmap = open(f'./trackeval/seqmap/{DATASET.lower()}/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        seqmap.write(f'{seq}\n')
    seqmap.close()
    print('tracking...')
    for seq in seqs:
        run(seq)
    print('evaluating...')
    evaluate(DATASET, SPLIT)