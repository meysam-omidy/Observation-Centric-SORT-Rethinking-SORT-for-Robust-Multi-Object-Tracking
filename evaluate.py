import trackeval
from utils import count_time
import numpy as np
import os

@count_time
def evaluate():
    trackers_to_eval = ['ocsort-self', 'oc-sort']
    # trackers_to_eval = 'cbiou'
    dataset = 'MOT17'

    eval_config = {'USE_PARALLEL': True,
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
                    'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': '../../.Datasets/MOT17/train/',
                        'TRACKERS_FOLDER': 'outputs',
                        'OUTPUT_FOLDER': None,
                        'TRACKERS_TO_EVAL': trackers_to_eval,
                        'CLASSES_TO_EVAL': ['pedestrian'],
                        'BENCHMARK': dataset if 'MOT' in dataset else 'MOT17',
                        # 'SPLIT_TO_EVAL': 'val',
                        'INPUT_AS_ZIP': False,
                        'PRINT_CONFIG': False,
                        'DO_PREPROC': True,
                        'TRACKER_SUB_FOLDER': '',
                        'OUTPUT_SUB_FOLDER': '',
                        'TRACKER_DISPLAY_NAMES': None,
                        'SEQMAP_FOLDER': None,
                        'SEQMAP_FILE': './trackeval/seqmap/%s/custom.txt' % dataset.lower(),
                        # 'SEQMAP_FILE': './trackeval/seqmap/%s/val.txt' % dataset.lower(),
                        'SEQ_INFO': None,
                        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                        'SKIP_SPLIT_FOL': True}


    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    os.makedirs('results', exist_ok=True)

    for tracker_to_eval in trackers_to_eval:

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
        
        file = open(f'results/{tracker_to_eval}-results.txt', 'w')
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
        file.close()

if __name__ == '__main__':
    evaluate()