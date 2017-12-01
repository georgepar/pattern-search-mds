import sys
sys.path.append('../')

from sacred import Experiment
from sacred.observers import MongoObserver

import multidimensional
import multidimensional.common
import multidimensional.mds
import multidimensional.point_filters
import multidimensional.radius_updates

import config

ex = Experiment("3000 vectors - 10 -> 3 dimensions - Random Data - Old version")
ex.observers.append(MongoObserver.create(
    url=config.SACRED_MONGO_URL,
    db_name=config.SACRED_DB
))


@ex.config
def cfg():
    num_vectors = 3000
    start_dim = 10
    target_dim = 3
    patience = 1
    max_turns = 10000
    error_barrier = 1e-20
    starting_radius = 0.1
    point_filter = multidimensional.point_filters.GSDFilter()
    radius_update = multidimensional.radius_updates.LinearRadiusDecrease()


@ex.automain
def experiment(num_vectors,
               start_dim,
               target_dim,
               patience,
               max_turns,
               error_barrier,
               starting_radius,
               point_filter,
               radius_update,
               _run):
    xs, _ = multidimensional.common.instance(num_vectors, start_dim)
    m = multidimensional.mds.MDS(
        target_dim,
        point_filter,
        radius_update,
        patience=patience,
        max_turns=max_turns,
        error_barrier=error_barrier,
        starting_radius=starting_radius,
        keep_history=True,
    )
    m.fit(xs)
    for i, error in enumerate(m.history['error']):
        _run.log_scalar('mds.mse.error', error, i + 1)
    for i, radius in enumerate(m.history['radius']):
        _run.log_scalar('mds.step', radius, i + 1)
    return m.history['error'][-1]
    # for i, xs in enumerate(m.history['xs_files']):
    #     _run.add_artifact(xs, name='xs_{}'.format(i + 1))
