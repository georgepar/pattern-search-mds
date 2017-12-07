import json
import sys

import metrics


def main():
    filename = sys.argv[1]
    basename = filename.split('.')[0]
    plot_file = '{}.png'.format(basename)
    description = ' '.join(basename.split('_'))
    with open(filename, 'r') as fd:
        exp_ids = json.load(fd)

    stored_metrics = metrics.StoredMetrics(exp_ids, description)
    stored_metrics.plot('Radius', plot_file)

if __name__ == '__main__':
    main()
