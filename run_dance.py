import os
import pickle
import numpy as np
from ocsort import OCSORTTracker
from track import Track
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED, TrackState
from evaluate_dance import evaluate
from utils import count_time

@count_time
def run(split):
    # seqs = ['MOT17-02-FRCNN', ]
    # seqs = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ]
    # seqs = [x.split('.')[0] for x in os.listdir('detections/ocsort_x_dance')]
    seqs = os.listdir(f'../../.Datasets/DanceTrack/{split}/')
    # seqs = ['dancetrack0069']



    os.makedirs('outputs/ocsort-self', exist_ok=True)

    seqmap = open('./trackeval/seqmap/dancetrack/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        print(seq)
        seqmap.write(f'{seq}\n')
        file = open(f'outputs/ocsort-self/{seq}.txt', 'w')
        # detections = np.loadtxt(f'detections/gt/{seq}.txt', delimiter=',')
        detections = np.loadtxt(f'detections/ocsort_x_dance/{seq}.txt', delimiter=',')
        gt_dets_file = np.loadtxt(f'../../.Datasets/DanceTrack/{split}/{seq}/gt/gt.txt', delimiter=',')

        # cbiou = CBIOUTracker()
        tracker = OCSORTTracker({
            # 'max_age': 150
            'association_speed_direction_coefficient': 0.2
        })

        for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
            # gt_dets, dets = gt_dets_file[gt_dets_file[:,0] == frame_number][:, 1:6], detections[int(frame_number)][:, :5]
            dets = detections[detections[:, 0] == frame_number][:, 1:]
            tracker.update(dets)
            for track in Track.get_tracks(outputs=True):
                file.write(f'{track.mot_format}\n')
            # if frame_number == 10:
            #     break
        file.close()
    seqmap.close()



if __name__ == '__main__':
    split = 'val'
    print('tracking...')
    run(split)
    print('evaluating...')
    evaluate(split)