import trackeval
from utils import count_time
import numpy as np
import os

DATASETS_DIR = 'C:/Projects/.Datasets/'

# Metrics shown in the per-sequence comparison. higher_better=False for IDSW.
COMPARE_METRICS = [
    ('HOTA', True), ('AssA', True), ('DetA', True), ('IDF1', True),
    ('MOTA', True), ('IDSW', False),
]


def _seq_metrics(ped_node):
    """Pull scalar metrics from res[...][tracker][seq]['pedestrian']."""
    def m(fam, key):
        v = ped_node[fam][key]
        v = np.mean(v) if getattr(v, 'ndim', 0) else v   # HOTA/AssA/DetA are per-alpha arrays
        return float(v.item() if hasattr(v, 'item') else v)
    return {
        'HOTA': m('HOTA', 'HOTA'), 'AssA': m('HOTA', 'AssA'), 'DetA': m('HOTA', 'DetA'),
        'IDF1': m('Identity', 'IDF1'), 'MOTA': m('CLEAR', 'MOTA'), 'MOTP': m('CLEAR', 'MOTP'),
        'IDSW': m('CLEAR', 'IDSW'), 'MT': m('CLEAR', 'MT'), 'ML': m('CLEAR', 'ML'),
    }


def _write_per_seq_comparison(metrics, trackers, seqs, path):
    """metrics[tracker][seqkey] -> dict. One table per metric: rows=seq, cols=trackers."""
    lines = []
    disp = {t: (t[:14]) for t in trackers}  # truncate names so 16-wide columns keep a gap
    for metric, higher_better in COMPARE_METRICS:
        lines.append('')
        lines.append('=' * (22 + 18 * len(trackers)))
        lines.append(f'{metric}   ({"higher" if higher_better else "lower"} is better;  * = best per sequence)')
        lines.append('=' * (22 + 18 * len(trackers)))
        header = f'{"sequence":<22}' + ''.join(f'{disp[t]:>16}' for t in trackers) + '   best'
        lines.append(header)
        lines.append('-' * len(header))
        for seq in seqs + ['COMBINED_SEQ']:
            vals = {t: metrics.get(t, {}).get(seq, {}).get(metric) for t in trackers}
            present = {t: v for t, v in vals.items() if v is not None}
            best_t = None
            if present:
                best_t = (max if higher_better else min)(present, key=present.get)
            row = f'{seq:<22}'
            for t in trackers:
                v = vals[t]
                if v is None:
                    cell = 'n/a'
                else:
                    cell = f'{int(v)}' if metric == 'IDSW' else f'{v:.4f}'
                    if t == best_t:
                        cell += '*'
                row += f'{cell:>16}'
            row += f'   {disp.get(best_t, "-")}'
            lines.append(row)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

@count_time
def evaluate(dataset, split, trackers_to_eval=None, datasets_dir=None):
    if trackers_to_eval is None:
        trackers_to_eval = ['ocsort-wbrt-final-baseline', 'ocsort-wbrt-learned-assoc-final', 'oc-sort', 'conf-r-rerun']
        # trackers_to_eval = ['ocsort-self', 'ocsort-self-v', 'oc-sort', 'ocsort-self-wbrt', 'official-ocsort-yoloxx']
        # trackers_to_eval = ['ocsort-self', 'ocsort-self-v', 'oc-sort', 'ocsort-self-transformer', 'ocsort-self-wbrt']
    if datasets_dir is None:
        datasets_dir = DATASETS_DIR

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

    dataset_config = {'GT_FOLDER': f'{datasets_dir}/{dataset}/{split}/',
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
    os.makedirs('results/per_seq', exist_ok=True)

    # ---- per-sequence metrics for every tracker (+ per-tracker CSV) ----
    per = {}  # per[tracker][seqkey] = metrics dict
    seq_names = None
    for tracker_to_eval in trackers_to_eval:
        tres = res['MotChallenge2DBox'].get(tracker_to_eval, {})
        per[tracker_to_eval] = {}
        for seqkey, node in tres.items():
            if 'pedestrian' not in node:
                continue
            per[tracker_to_eval][seqkey] = _seq_metrics(node['pedestrian'])
        # sequence list (exclude the aggregate), from whichever tracker has them
        names = sorted(k for k in per[tracker_to_eval] if k != 'COMBINED_SEQ')
        if names and seq_names is None:
            seq_names = names
        # per-tracker per-sequence CSV
        cols = ['HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'MOTP', 'IDSW', 'MT', 'ML']
        with open(f'results/per_seq/{tracker_to_eval}.csv', 'w', encoding='utf-8') as f:
            f.write('sequence,' + ','.join(cols) + '\n')
            for seqkey in (names + ['COMBINED_SEQ']):
                mvals = per[tracker_to_eval].get(seqkey, {})
                f.write(seqkey + ',' + ','.join(
                    ('' if mvals.get(c) is None else
                     (f'{int(mvals[c])}' if c == 'IDSW' else f'{mvals[c]:.6f}')) for c in cols) + '\n')
    seq_names = seq_names or []

    # ---- cross-tracker per-sequence comparison table ----
    _write_per_seq_comparison(per, trackers_to_eval, seq_names,
                              f'results/per_seq_comparison_{dataset.lower()}.txt')

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
    # evaluate('MOT20', 'val')
    # evaluate('MOT17', 'val')
    evaluate('DanceTrack', 'val')