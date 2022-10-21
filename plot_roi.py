#!/usr/bin/env python3

import argparse
import atexit
from multiprocessing import Process, Queue
from multiprocessing.connection import Listener, Client
from threading import Thread
import time
import queue


SERVER_HOST = 'localhost'
SERVER_PORT = 12481

MAX_LIFETIME_S = 60 * 60 * 2

def main():
    parser = argparse.ArgumentParser(description='Reads and plots ROI stats from cached'
        ' responses written by al_analysis.py. Can also analyze fresh data, and compare'
        ' this to cached responses.'
        # importing this would cause too much other stuff to be imported and slow down
        # calls just communicating to already running server
        #f'{ij_roi_responses_cache}'
    )
    # TODO TODO still check it is specified in case where -d not passed
    parser.add_argument('roi_strs', nargs='*', help='ROI names (e.g. DM5, 3-30/1/0)')
    parser.add_argument('-d', '--analysis-dir', help='If passed, analyze data from this'
        ' directory, rather than loading data from al_analysis.py cached responses.'
        #f' directory, rather than loading data from {ij_roi_responses_cache}.'
    )
    # TODO store as int(s)
    # TODO change to roi_indices (tho would i then need to select all from ROI manager
    # rather than overlay? might be tricky)
    parser.add_argument('-i', '--roi-index', type=int, help='The index of the ROI to '
        'analyze. Only relevant when also passing -d/--analysis-dir.'
    )
    parser.add_argument('-r', '--roiset-path', help='Path to the RoiSet.zip to load '
        'ImageJ ROIs from. Only relevant in -d/--analysis-dir case.'
    )
    # TODO maybe another option to show everything that did NOT match substring as well
    # (matching cached -> specific indexed ROI not in cache -> NON-matching cached)
    # TODO rename to something w/ "cache" or "analyzed" in it (-> use "compare" for
    # flag indicating we want to compare to currently analyzed data, from previous calls
    # while server is still active)
    parser.add_argument('-n', '--no-compare', action='store_true', help='Only plot data'
        ' from -d/--analysis_dir, rather than also plotting relevant cached data above '
        'it.'
    )
    parser.add_argument('-p', '--pairs', action='store_true', help='Also plots'
        'pair experiment data, when available. Excluded from plots by default.'
    )
    # TODO i just would still want to see neighboring concentrations, if available...
    # (e.g. -5 vs -6 2,3-butanedione, after diagnostic conc change)
    parser.add_argument('-o', '--other-odors', action='store_true', help='Also plots'
        'data from odors not done for the fly referenced by -d/--analysis-dir, when '
        'available. Excluded from plots by default.'
    )
    parser.add_argument('-a', '--add', action='store_true', help='Updates current plot '
        'to include data from newly specified ROI. Only relevant in -d/--analysis-dir '
        'case. Otherwise, a new plot is made in parallel.'
    )
    args = parser.parse_args()

    try:
        # TODO maybe i want to use something other than 'AF_INET'? is there a default?
        listener = Listener((SERVER_HOST, SERVER_PORT), 'AF_INET')

    except OSError:
        listener = None

    # will it work to do this from the same script? need a separate thread/process
    # maybe?
    client = Client((SERVER_HOST, SERVER_PORT), 'AF_INET')
    client.send(args)
    if listener is None:
        return

    start_time_s = time.time()

    # Only importing these *after* client would have exited, so those invocations don't
    # need to suffer long import time.
    from al_analysis import init_logger
    from plot_roi_util import load_and_plot

    log = init_logger(__name__, __file__)

    arg_queue = Queue()

    def get_client_args():
        # This is the call that can hang, hence why I'm wrapping this call in a Process
        client = listener.accept()

        client_args = client.recv()
        client.close()

        arg_queue.put(client_args)

    def get_cliet_args_loop():
        while True:
            get_client_args()

    p = Process(target=get_cliet_args_loop)
    p.start()

    # This, rather than just the same call after the while loop, avoided some hanging
    # caused by exceptions in load_and_plot.
    atexit.register(p.terminate)

    # TODO how to decide when to shut the server down? do we?
    # can register something at imagej exit (inside imagej config) to kill any?
    # any nice way to have python check if (corresponding) imagej process is alive?

    # TODO maybe only go into this loop if a certain flag is set / if we are analyzing
    # new data
    while time.time() - start_time_s < MAX_LIFETIME_S:
        try:
            client_args = arg_queue.get_nowait()
        except queue.Empty:
            continue

        load_and_plot(client_args)


if __name__ == '__main__':
    main()

