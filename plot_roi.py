#!/usr/bin/env python3

import argparse
from multiprocessing.connection import Listener, Client
from threading import Thread
import time
import queue

SERVER_HOST = 'localhost'
SERVER_PORT = 12481

#MAX_LIFETIME_S = 60 * 60 * 2
MAX_LIFETIME_S = 10


def main():
    parser = argparse.ArgumentParser(description='Reads and plots ROI stats from cached'
        ' responses written by al_analysis.py. Can also analyze fresh data, and compare'
        ' this to cached responses.'
        # importing this would cause too much other stuff to be imported and slow down
        # calls just communicating to already running server
        #f'{ij_roi_responses_cache}'
    )
    # TODO TODO still check it is specified in case where -a not passed
    parser.add_argument('roi_strs', nargs='*', help='ROI names (e.g. DM5, 3-30/1/0)')
    parser.add_argument('-a', '--analysis-dir', help='If passed, analyze data from this'
        ' directory, rather than loading data from al_analysis.py cached responses.'
        #f' directory, rather than loading data from {ij_roi_responses_cache}.'
    )
    # TODO store as int(s)
    # TODO change to roi_indices (tho would i then need to select all from ROI manager
    # rather than overlay? might be tricky)
    parser.add_argument('-i', '--roi-index', type=int, help='The index of the ROI to '
        'analyze. Only relevant when also passing -a/--analysis-dir.'
    )
    parser.add_argument('-r', '--roiset-path', help='Path to the RoiSet.zip to load '
        'ImageJ ROIs from. Only relevant in -a/--analysis-dir case.'
    )
    # TODO maybe another option to show everything that did NOT match substring as well
    # (matching cached -> specific indexed ROI not in cache -> NON-matching cached)
    # TODO rename to something w/ "cache" or "analyzed" in it (-> use "compare" for
    # flag indicating we want to compare to currently analyzed data, from previous calls
    # while server is still active)
    parser.add_argument('-n', '--no-compare', action='store_true', help='Only plot data'
        ' from -a/--analysis_dir, rather than also plotting relevant cached data above '
        'it.'
    )
    parser.add_argument('-p', '--pairs', action='store_true', help='Also plots'
        'pair experiment data, when available. Excluded from plots by default.'
    )
    # TODO i just would still want to see neighboring concentrations, if available...
    # (e.g. -5 vs -6 2,3-butanedione, after diagnostic conc change)
    parser.add_argument('-o', '--other-odors', action='store_true', help='Also plots'
        'data from odors not done for the fly referenced by --analysis-dir, when '
        'available. Excluded from plots by default.'
    )
    args = parser.parse_args()

    try:
        # TODO maybe i want to use something other than 'AF_INET'? is there a default?
        listener = Listener((SERVER_HOST, SERVER_PORT), 'AF_INET')

    except OSError:
        listener = None

    # TODO maybe refactor so all plot calls can be killed by a client connection
    # (put actual plotting in a subprocess?)
    # was trying stuff from https://stackoverflow.com/questions/28269157
    # to get it working using matplotlibs non-blocking calls

    # will it work to do this from the same script? need a separate thread/process
    # maybe?
    client = Client((SERVER_HOST, SERVER_PORT), 'AF_INET')
    client.send(args)
    if listener is None:
        return

    #plt.ion()
    #plt.show()

    start_time_s = time.time()

    # Only importing these *after* client would have exited, so those invocations don't
    # need to suffer long import time.
    from al_analysis import init_logger
    from plot_roi_util import load_and_plot

    log = init_logger(__name__, __file__)

    # TODO maybe try https://stackoverflow.com/questions/57817955
    # to implement a listener timeout?
    # or re-implement w/ something like https://stackoverflow.com/questions/50031613 ?

    # TODO how to decide when to shut the server down? do we?
    # can register something at imagej exit (inside imagej config) to kill any?
    # any nice way to have python check if (corresponding) imagej process is alive?

    arg_queue = queue.Queue()
    def get_client_args():
        print('accept', flush=True)
        # This is the call that can hang, hence why I'm wrapping this call in a Thread
        client = listener.accept()

        print('recv', flush=True)
        client_args = client.recv()
        print('close', flush=True)
        client.close()

        arg_queue.put(client_args)

        print('', flush=True)

    # TODO maybe only go into this loop if a certain flag is set / if we are analyzing
    # new data
    while time.time() - start_time_s < MAX_LIFETIME_S:

        #plt.pause(0.001)

        # TODO TODO don't make a new thread if the old one is alive? will gc handle the
        # dereferenced threads at least?

        # TODO probably switch these to using multiprocessing.Process / similar, as no
        # way to kill threads at the end, and even after we exceed MAX_LIFETIME_S, it
        # seems we are waiting on one (or more) threads to finish an accept call that
        # will never come...

        t = Thread(target=get_client_args)
        print('start', flush=True)
        t.start()
        print('join', flush=True)
        # TODO how much time does it actually need to reliably populate the queue w/ the
        # client args?
        t.join(0.1)

        # don't think this actually matters (assuming old threads get cleaned up
        # sufficiently just by dereferencing them...)
        print('is_alive', flush=True)
        if t.is_alive():
            print('timeout', flush=True)


        try:
            client_args = arg_queue.get_nowait()
        except queue.Empty:
            print('queue was empty', flush=True)
            continue

        print('load_and_plot', flush=True)
        load_and_plot(client_args)


if __name__ == '__main__':
    main()

