import numpy as np
from ocsort import OCSORTTracker
from evaluate import evaluate
from utils import count_time
import configparser
import os


DATASET = 'MOT17'
SPLIT = 'train'
SEQS = seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]
# SEQS = None
DETECTION_FOLDER = 'bytetrack_x_mot17'

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