import numpy as np
from ocsort import OCSORTTracker
from evaluate import evaluate
from utils import count_time
import configparser
import os
import concurrent.futures

# DATASET = 'MOT20'
DATASET = 'MOT17'
# DATASET = 'DanceTrack'
SPLIT = 'val'
# SEQS = seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]
# SEQS = None
SEQS = ['MOT17-04-FRCNN']
# SEQS = ['MOT20-01']
# SEQS = ['dancetrack0079']
# DETECTION_FOLDER = 'ocsort_x_mot20'
DETECTION_FOLDER = 'bytetrack_x_mot17'
# DETECTION_FOLDER = 'ocsort_x_dance'
DATASETS_DIR = 'D://Projects/.Datasets/'
# DATASETS_DIR = '../../../.Datasets'

@count_time
def run(seq):
    print(seq)
    # os.makedirs('outputs/ocsort-self', exist_ok=True)
    file = open(f'outputs/ocsort-self/{seq}.txt', 'w')
    detections = np.loadtxt(f'detections/{DETECTION_FOLDER}/{seq}.txt', delimiter=',')
    config = configparser.ConfigParser()
    config.read(f'{DATASETS_DIR}/{DATASET}/{SPLIT}/{seq}/seqinfo.ini')
    tracker = OCSORTTracker({
        'image_width': config['Sequence']['imWidth'],
        'image_height': config['Sequence']['imHeight'],
        'association_speed_direction_coefficient': 0,
        'use_byte': True,
        'reupdate_type': 'constant',
        'reupdate_constant_weight': '0.8',
        'motion': {
            'enabled': True,
            'model_type': 'transformer_learned',
            'weights_path': 'motion_model_weights/transformer_learned_qr.pth',
            'use_kalman': True,
            'kalman_fusion_blend': 0.5,
            # 'model_type': 'lstm_learned',
            # 'weights_path': 'motion_model_weights/phase2_lstm_learned.pth',
        },
    })
    for frame_number in range(1, int(config['Sequence']['seqLength']) + 1):
        print(frame_number)
        dets = detections[detections[:, 0] == frame_number][:, 1:]
        tracker.update(dets)
        for output in tracker.get_outputs():
            file.write(f'{output}\n')
    file.close()


if __name__ == '__main__':
    if SEQS:
        seqs = SEQS
    else:
        seqs = os.listdir(f'{DATASETS_DIR}/{DATASET}/{SPLIT}/')
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