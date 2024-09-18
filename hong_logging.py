# TODO probably move all this to hong2p
"""
Logging shared by al_analysis.py and plot_roi[_util].py
"""

import atexit
import logging
from pathlib import Path
import shlex
import sys
from typing import Optional

# NOTE: this 3rd party package doesn't work on Windows
# (and may have issues in some other specific circumstances, such as if there are
# threads involved too)
from multiprocessing_logging import install_mp_handler

# TODO profile extra time this import adds -> delete/type-checking-guard if it's
# anything >.2s or so
from hong2p.types import Pathlike


# NOTE: currently (at least up to 2023-12-12) only logging initial command and any
# errors. NOT EVEN COMPATIBLE WITH ANY EXISTING logging calls
# TODO TODO try to make compatible w/ existing logging calls (avoid need to return
# logger object)
#
# TODO refactor to use the global (module specific?) logger, to enable for nested code
# w/o passing logger in (do i ever actually use the logger this returns? seems no...)
# TODO TODO this also modifies default logging handler for whatever calls it, right?
# [assuming being called correctly, as in docstring here])
# (NO, i think it just modifies excepthook and does the initial log of argv...)
# TODO delete multiprocessing kwarg if i don't ever really benefit from disabling it
def init_logger(module_name: Optional[str], script_path: Pathlike, *,
    multiprocessing: bool = True, verbose: bool = False) -> logging.Logger:
    """Logs sys.argv and future uncaught exceptions to file + stderr.

    Logs are saved to a logs/ directory (created if needed) at the same level as
    `script_path`, named with the same prefix (i.e. without file extension) as the
    script.

    Currently returns logger to log to same file, though I'd like if I could change to
    allowing future `logging.` calls to work (and recursively, w/ logging calls in
    any called modules).

    >>> log = init_logger(__name__, __file__)

    If `verbose=True`, will print to stdout which file we are logging to.
    """
    # TODO rename to logger_name or name if i'm also sometimes hardcoding it (as in
    # plot_roi.py now)
    # TODO delete (and delete module_name arg), if root logger approach below works
    logger = logging.getLogger(module_name)
    del module_name

    # TODO under what circumstances is a logger inheriting the config of it's
    # "parent"? is plot_roi_util.py logger (what getLogger returns) going to inherit
    # config in plot_roi.py?) if so, detect and don't reset config in those cases?
    # (just hardcoding logger name to same in both for now, so could delete this)

    script_path = Path(script_path).resolve()

    log_dir = script_path.parent / 'logs'
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / f'{script_path.stem}.log'
    del script_path

    if verbose:
        # using stderr, to be consistent w/ where stream_handler below will output
        print(f'logging to {log_path}', file=sys.stderr)

    # TODO some way to read the log that lets me easily look at stuff from just single
    # PIDs? feel like i probably don't want to save each run to a separate file, but
    # maybe?

    handler = logging.FileHandler(log_path)

    # TODO might also want:
    # name,filename|module,funcName,lineno,processName,thread,threadName
    # see: https://docs.python.org/3/library/logging.html#logrecord-attributes
    # TODO when is process not available (docs say "if available")? relevant to any of
    # my uses?
    #
    # %(name)s might be useful later if i actually get recursive logging to work here,
    # but not working/testing now.
    FORMAT = '%(asctime)s PID=%(process)d %(levelname)-8s %(message)s'
    formatter = logging.Formatter(fmt=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # TODO TODO probably take this level as a kwarg, esp if i can't set different levels
    # for diff handlers (w/ StreamHandler having less than DEBUG, maybe default WARNING)
    logger.setLevel(logging.DEBUG)

    # might not need this explicitly (maybe fine that we have set logger level above?)
    # see:
    # https://stackoverflow.com/questions/11111064
    # https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/
    handler.setLevel(logging.DEBUG)

    logger.addHandler(handler)

    # TODO TODO also log python executable path (hopefully can determine any conda/venv
    # from this) by default
    # TODO log all environment variables? ones i use in here / hong2p (might be annoying
    # to implement, and could diverge...)?

    # To also print to stderr (default behavior of StreamHandler. should not be any
    # output to stdout.)
    stream_handler = logging.StreamHandler()
    # DEBUG < INFO < WARN < ERROR < CRITICAL
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # TODO TODO also [an option to] log git commit/remote[/maybe any unstaged changes?]
    # of enclosing repo. maybe even save patch for unstaged changes to log[/separate]
    # dir?

    # TODO maybe log parent PID? not sure i actually care, but could help clarify chain
    # that leads to stuff getting run, when invoked from imagej

    # TODO test this actually makes sense w/ both venv and conda
    #
    # mainly to have a record of whether a particular environment's python is being used
    # to run the script. should work w/ venv and conda
    py_exec_msg = f'python={sys.executable}'
    py_exec = Path(sys.executable)
    if py_exec.is_symlink():
        # NOTE: this will follow any symlinks in target too, e.g.
        # $ ls -l venv/bin/python3
        # ... venv/bin/python3 -> /usr/bin/python3
        # but if /usr/bin/python3 points to /usr/bin/python3.8, then the latter will
        # be logged as target below (probably what I want anyway).
        py_exec_msg = f'{py_exec_msg} (-> {py_exec.resolve()})'

    logger.debug(py_exec_msg)

    cmd_str = shlex.join(sys.argv)
    logger.debug(cmd_str)


    def log_clean_exit():
        # TODO assert this is only registered once / run once per PID*? (prob no need?)
        logger.debug('exiting cleanly')

    atexit.register(log_clean_exit)


    def handle_exception(exc_type, exc_value, exc_traceback):
        # either way we are going to exit, and don't want to log 'exiting cleanly'
        atexit.unregister(log_clean_exit)

        # bdb.BdqQuit doesn't seem to currently get logged here, though I'm not sure why
        #
        # actually, it seems any exception after I have been in ipdb (e.g. despite 'c'
        # to continue to completion), isn't logged.
        #
        # TODO possible to get those errors to be logged? (still true after not special
        # causing KeyboardInterrupt as much?)

        if issubclass(exc_type, KeyboardInterrupt):
            # TODO any way to see tracebacks for things other than the single parent
            # process (e.g. plot process or plot_roi.py subprocess queueing client
            # args?)? maybe not? even matter?
            #
            # NOTE: if I want to see traceback when pkill-ing plot_roi.py (or in
            # general), need to use SIGINT rather than default SIGTERM (at least without
            # other hacky stuff), so:
            # `pkill -SIGINT -f '/plot_roi'`
            logger.debug('killed via KeyboardInterrupt/SIGINT',
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        else:
            logger.critical('Uncaught exception',
                exc_info=(exc_type, exc_value, exc_traceback)
            )

    sys.excepthook = handle_exception

    # TODO TODO this work if the subprocess spawns it's own processes?
    # (as plot_roi_util.py does for plotting)
    # TODO even work without only using one Pool?
    if multiprocessing:
        # TODO TODO this only work w/ root logger? if so, may need to delete
        install_mp_handler()

    return logger

