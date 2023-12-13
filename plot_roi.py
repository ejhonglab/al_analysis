#!/usr/bin/env python3

import argparse
import atexit
from multiprocessing import Process, Queue
from multiprocessing.connection import Listener, Client
import time
import queue

# Factored init_logger into this minimal module (outside al_analysis.py), to avoid
# major import time costs associated with importing al_analysis, such that we can import
# this unconditionally up here (and log stuff doing server/client setup)
from hong_logging import init_logger


SERVER_HOST = 'localhost'
SERVER_PORT = 12481

MAX_LIFETIME_S = 60 * 60 * 2


def main():
    # TODO TODO TODO make sure this is actually set up such that both server and clients
    # log all errors (esp fatal ones) to a sensible place (maybe one log file, as long
    # as info to distinguish PIDs and server/client easily is added)
    # (so that when killing a hung process, i can see where it is hung, to fix it)

    # TODO might be useful to use filename in log format (now that i'm using same exact
    # config in plot_roi_util.py)?
    #
    # Just gonna hardcode this and then use that in plot_roi_util.py for now
    log = init_logger('plot_roi', __file__)

    ## TODO replace log w/ just logging (now that init_logger is changing root logger)?
    #log = init_logger(__name__, __file__)

    # TODO TODO make a fn for printing all PIDs in tree under the parent python process
    # -> intersperse below to figure out which calls are actually making new processes
    # (or otherwise figure out) (is it just Process? wb Queue/Listener/Client?)

    parser = argparse.ArgumentParser(description='Reads and plots ROI stats from cached'
        ' responses written by al_analysis.py. Can also analyze fresh data, and compare'
        ' this to cached responses.'
        # importing this would cause too much other stuff to be imported and slow down
        # calls just communicating to already running server
        #f'{ij_roi_responses_cache}'
    )
    # TODO still check it is specified in case where -d not passed
    parser.add_argument('roi_strs', nargs='*', help='ROI names (e.g. DM5, 3-30/1/0)')
    parser.add_argument('-d', '--analysis-dir', help='If passed, analyze data from this'
        ' directory, rather than loading data from al_analysis.py cached responses.'
        #f' directory, rather than loading data from {ij_roi_responses_cache}.'
    )
    parser.add_argument('-i', '--roi-index', type=int, help='The index of the ROI to '
        'analyze. Only relevant when also passing -d/--analysis-dir.'
    )
    parser.add_argument('-r', '--roiset-path', help='Path to the RoiSet.zip to load '
        'ImageJ ROIs from. Only relevant in -d/--analysis-dir case.'
    )
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
    # TODO TODO add CLI flag that alters behavior so no plotting is done but index of
    # odor with max response is printed (-> use in imagej_macros to go to index)

    # TODO add parameter to specify within which concentration range stuff should be
    # considered
    parser.add_argument('-H', '--hallem', action='store_true',
        help='Also plots Hallem responses to glomeruli, sorted by correlation to the '
        'ROI from -d/--analysis-dir. Currently only considers odors that match the '
        'Hallem concentration exactly (-3).'
    )

    parser.add_argument('-b', '--debug', action='store_true',
        help='Should prevent calls within plot_roi_util.py from running through '
        'multiprocessing.Process, to be able to use a debugger within those calls.'
    )
    args = parser.parse_args()

    # TODO try to add arg above and refactor below so that i can run everything w/o
    # multiprocessing, for debugging of plot_roi_util (can't currently use debugger in
    # there...)

    log.debug(f'trying to start listener on port {SERVER_PORT}')
    try:
        # TODO log that we are starting this and what PID is
        # TODO maybe i want to use something other than 'AF_INET'? is there a default?
        listener = Listener((SERVER_HOST, SERVER_PORT), 'AF_INET')

        log.debug('started listener')

    except OSError:
        log.debug('could not start listener (one probably already bound to port)!')
        listener = None

    log.debug(f'starting client on port {SERVER_PORT}')
    # will it work to do this from the same script? need a separate thread/process
    # maybe?
    client = Client((SERVER_HOST, SERVER_PORT), 'AF_INET')

    # TODO may not want to actually log args
    log.debug(f'sending args {args} via client')
    client.send(args)

    # TODO does this imply they've been successfully received???
    log.debug('done sending args')

    if listener is None:
        log.debug('this process was a client only. exiting (having sent args to '
            'listener)!'
        )
        # TODO maybe disable stream handler init_logger also adds tho?  want in
        # al_analysis.py, but probably don't want this printed? or just use debug level?
        return

    start_time_s = time.time()

    # TODO add a comment saying how i measured import time again + how to do rest of
    # profiling (-> do again. i feel like things are pretty slow now)
    #
    # Only importing these *after* client would have exited, so those invocations don't
    # need to suffer long import time (importing al_analysis imports a lot of other
    # stuff, which takes maybe 1-3 seconds).
    from plot_roi_util import load_and_plot

    arg_queue = Queue()

    def get_client_args():
        # TODO these even need multiprocessing_logging to work correctly? do i ever
        # actually have two things writing to the file at once?
        # (feel like not, or at least maybe only if i also log in plot subprocesses in
        # plot_roi_util.py?)
        # (delete that dependency if not)
        log.debug('get_client_args: calling listener.accept()')
        # This is the call that can hang, hence why I'm wrapping this call in a Process
        client = listener.accept()

        log.debug('get_client_args: calling client.recv()')
        client_args = client.recv()

        log.debug('get_client_args: got client_args from client.recv()')
        # TODO need to log before this one? prob not
        client.close()

        # TODO need to log before this one? prob not
        arg_queue.put(client_args)

    def get_client_args_loop():
        # TODO TODO will these logging calls above (ini get_client_args) even work
        # (since this is called from Process? need to change global logging settings /
        # re-init logging here [or in a new target that includes that?]?)
        try:
            while True:
                get_client_args()

        # TODO test this even works / delete if not
        # https://stackoverflow.com/questions/34506638
        # TODO and does p.terminate also trigger this?
        finally:
            # not sure i even care to log this (in any case)
            #
            # (i kinda just wanted to see which step of get_client_args it's add when
            # killed, especially if that's the reason the overall system sometimes
            # hangs)
            log.debug('get_client_args_loop seems to have been interrupted')

    p = Process(target=get_client_args_loop)
    log.debug('listener: starting process to queue incoming client args')
    # TODO can i get PID from `p`? do i need to start it first?  just get logging
    # working correctly inside it, then use that to show PID and other stuff (what is
    # sys.argv inside there tho? don't want to duplicate from parent...)
    p.start()

    # TODO wrap p.terminate in something that also log.debug-s?
    #
    # This, rather than just the same call after the while loop, avoided some hanging
    # caused by exceptions in load_and_plot.
    atexit.register(p.terminate)

    # TODO can register something at imagej exit (inside imagej config) to kill any?
    # any nice way to have python check if (corresponding) imagej process is alive?

    # TODO atexit call to log PID and memory usage (for seeing which processes were
    # actually using memory when pkilled)
    # TODO also log current / max lifetime atexit, to see if any things are somehow
    # exceeding lifetime

    while time.time() - start_time_s < MAX_LIFETIME_S:
        try:
            client_args = arg_queue.get_nowait()
        except queue.Empty:
            continue

        # TODO keep printing args? OK format (pformat?)?
        log.debug(f'listener: calling load_and_plot with args={client_args}')

        load_and_plot(client_args)

        # TODO TODO is this causing issues? delete?
        # TODO how does breaking here prevent making a server though... does it?
        # hasn't that decision already been made? this comment out-of-date?
        # TODO TODO and why does this being not None indicate what the comment below is
        # saying?
        # Don't want to start a server unless we actually did io/compute that takes a
        # lot of time. All calls from ImageJ should have --analysis-dir specified.
        if args.analysis_dir is None:
            log.debug('listener: breaking because args.analysis_dir (from client args '
                'queue) was None'
            )
            break


if __name__ == '__main__':
    main()

