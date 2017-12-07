import sys

import pymongo
from bson.objectid import ObjectId
import matplotlib.pyplot as plt

import config


class StoredMetrics(object):
    def __init__(self, experiment_metrics_ids, experiment_description):
        self.experiment_description = experiment_description
        self.experiment_metrics_ids = {
            exp_name: {
                metric_name: ObjectId(metric_id)
                for metric_name, metric_id in exp_metrics.items()
            }
            for exp_name, exp_metrics in experiment_metrics_ids.items()
        }
        client = pymongo.MongoClient(config.SACRED_MONGO_URL)
        self.metrics_collection = client[config.SACRED_DB].metrics
        self.metrics = {}

    def retrieve(self):
        metrics = {
            exp_name: {
                metric_name: self.metrics_collection.find_one(
                    {'_id': metric_id})
                for metric_name, metric_id in exp_metrics.items()
            }
            for exp_name, exp_metrics in self.experiment_metrics_ids.items()
        }
        self.metrics = metrics
        return metrics

    def plot(self, metric_name, filename):
        if len(self.metrics) == 0:
            all_metrics = self.retrieve()
        else:
            all_metrics = self.metrics
        import pprint
        pprint.pprint(all_metrics)
        metrics = {
            exp_name: {
                'x': exp_metrics[metric_name]['steps'],
                'y': exp_metrics[metric_name]['values']
            }
            for exp_name, exp_metrics in all_metrics.items()
        }
        plt.hold(True)
        for exp_name, axes in metrics.items():
            label = '{}, Final Turn: {}, Final Error: {}'.format(
                exp_name, axes['x'][-1], axes['y'][-1])
            plt.plot(axes['x'], axes['y'], label=label)
        plt.title("Compare {} for {}".format(
            metric_name, self.experiment_description))
        plt.ylabel(metric_name)
        plt.xlabel('Epochs')
        plt.legend(loc='best')
        plt.savefig(filename)


