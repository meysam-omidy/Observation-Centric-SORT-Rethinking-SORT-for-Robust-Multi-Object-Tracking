from ocsort import OCSORTTracker
from track import Track, STATE_TRACKING
import trackeval
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import json
import os

def evaluate(seq):
    trackers_to_eval = ['ocsort-self', 'oc-sort']
    dataset = 'dancetrack'
    eval_config = {
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': '../outputs/error_log.txt',
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': False,
        'OUTPUT_EMPTY_CLASSES': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False
    }
    dataset_config = {
        'GT_FOLDER': '../../.Datasets/DanceTrack/train/',
        'TRACKERS_FOLDER': 'outputs',
        'OUTPUT_FOLDER': None,
        'TRACKERS_TO_EVAL': trackers_to_eval,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'dancetrack',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': '',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,
        'SEQMAP_FILE': f'./trackeval/seqmap/dancetrack/{seq}.txt',
        'SEQ_INFO': None,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': True
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)
    os.makedirs('ablation_results', exist_ok=True)
    for tracker_to_eval in trackers_to_eval:
        os.makedirs(f'ablation_results/{tracker_to_eval}', exist_ok=True)
        hota = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
        idf1 = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1'].item()
        mota = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA'].item()
        motp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTP'].item()
        assa = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
        deta = np.mean(res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()
        idsw = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['IDSW'].item()
        tp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_TP']
        fp = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_FP']
        fn = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['CLR_FN']
        mt = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MT'].item()
        ml = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['ML'].item()
        count = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Count']
        with open(f'ablation_results/{tracker_to_eval}/{seq}.json', 'w') as f:
            json.dump({
                'MOTA': mota,
                'MOTP': motp,
                'IDF1': idf1,
                'IDSW': idsw,
                'HOTA': hota,
                'FP': fp,
                'FN': fn,
                'IDS': count['IDs']
            }, f)

def run(seq):
    print(f'started {seq}')
    os.makedirs('ablations/ocsort-self', exist_ok=True)
    seqmap = open(f'./trackeval/seqmap/dancetrack/{seq}.txt', 'w')
    seqmap.write('name\n')
    seqmap.write(f'{seq}\n')
    seqmap.close()
    file = open(f'outputs/ocsort-self/{seq}.txt', 'w')
    detections = np.loadtxt(f'detections/ocsort_x_dance/{seq}.txt', delimiter=',')
    gt_dets_file = np.loadtxt(f'../../.Datasets/DanceTrack/train/{seq}/gt/gt.txt', delimiter=',')
    tracker = OCSORTTracker({
        'association_speed_direction_coefficient': 0
    })
    for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
        dets = detections[detections[:, 0] == frame_number][:, 1:]
        tracker.update(dets)
        for track in Track.get_tracks([STATE_TRACKING]):
            file.write(f'{track.mot_format}\n')
    file.close()
    evaluate(seq)
    print(f'finished {seq}')
    os.remove(f'./trackeval/seqmap/dancetrack/{seq}.txt')
    return


if __name__ == '__main__':
    for seq in os.listdir('../../.Datasets/DanceTrack/train/'):
        run(seq)
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     seqs = os.listdir('../../.Datasets/DanceTrack/train/')
    #     futures = [executor.submit(run, seq) for seq in ['dancetrack0001', 'dancetrack0002']]
    #     # futures = [executor.submit(run, seq) for seq in seqs]
    #     for future in as_completed(futures):
    #         future.result()