import numpy as np
import trackeval
from ocsort import OCSORTTracker
from utils import count_time
import configparser
import os
import json
import pickle

DATASET = 'DanceTrack'
SPLIT = 'train'
SEQS = None
DETECTION_FOLDER = 'ocsort_x_dance'
ABLATION_NAME = 'with_byte_and_config_pr'


def evaluate(dataset, split):
    trackers_to_eval = ['ocsort-self', 'oc-sort']
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
        'GT_FOLDER': f'../../.Datasets/{dataset}/{split}/',
        'TRACKERS_FOLDER': 'outputs',
        'OUTPUT_FOLDER': None,
        'TRACKERS_TO_EVAL': trackers_to_eval,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': dataset if 'MOT' in dataset else 'MOT17',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': True,
        'TRACKER_SUB_FOLDER': '',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,
        'SEQMAP_FILE': './trackeval/seqmap/%s/custom.txt' % dataset.lower(),
        'SEQ_INFO': None,
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': True
    }
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)
    os.makedirs(f'ablation_results/{ABLATION_NAME}/json', exist_ok=True)
    os.makedirs(f'ablation_results/{ABLATION_NAME}/raw', exist_ok=True)
    for tracker_to_eval in trackers_to_eval:
        os.makedirs(f'ablation_results/{ABLATION_NAME}/json/{tracker_to_eval}', exist_ok=True)
        os.makedirs(f'ablation_results/{ABLATION_NAME}/raw/{tracker_to_eval}', exist_ok=True)
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
        with open(f'ablation_results/{ABLATION_NAME}/json/{tracker_to_eval}/{seq}.json', 'w') as f:
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
        with open(f'ablation_results/{ABLATION_NAME}/raw/{tracker_to_eval}/{seq}.txt', 'w') as file:
            file.write(f'MOTA:    {mota}\n')
            file.write(f'MOTP:    {motp}\n')
            file.write(f'TP:      {tp}\n')
            file.write(f'FP:      {fp}\n')
            file.write(f'FN:      {fn}\n')
            file.write(f'IDSW:    {idsw}\n')
            file.write(f'IDF1:    {idf1}\n')
            file.write(f'MT:      {mt}\n')
            file.write(f'ML:      {ml}\n')
            file.write(f'HOTA:    {hota}\n')
            file.write(f'ASSA:    {assa}\n')
            file.write(f'DETA:    {deta}\n')
            file.write('\n')
            file.write('\n')
            count = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Count']
            for key in count.keys():
                file.write(f'{key}{" "*(11 - len(key))}{count[key]}\n')
            file.write('\n')
            file.write('\n')
            identity = res['MotChallenge2DBox'][tracker_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']
            for key in identity.keys():
                file.write(f'{key}{" "*(8 - len(key))}{identity[key].item()}\n')

@count_time
def run(seq):
    print(seq)
    os.makedirs(f'ablation_results/{ABLATION_NAME}/log/ocsort-self', exist_ok=True)
    tracker = OCSORTTracker({
        'log_path': f'ablation_results/{ABLATION_NAME}/log/ocsort-self/{seq}.log',
        'use_byte': True
    })
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
    os.makedirs(f'ablation_results/{ABLATION_NAME}/pickle/ocsort-self', exist_ok=True)
    with open(f'ablation_results/{ABLATION_NAME}/pickle/ocsort-self/{seq}.pickle', 'wb') as f:
        pickle.dump(tracker, f)



if __name__ == '__main__':
    if SEQS:
        seqs = SEQS
    else:
        seqs = os.listdir(f'../../.Datasets/{DATASET}/{SPLIT}/')
    # seqs = ['dancetrack0033']
    os.makedirs('ablations/ocsort-self', exist_ok=True)
    for seq in seqs:
        seqmap = open(f'./trackeval/seqmap/{DATASET.lower()}/custom.txt', 'w')
        seqmap.write('name\n')
        seqmap.write(f'{seq}\n')
        seqmap.close()
        run(seq)
        evaluate(DATASET, SPLIT)
    