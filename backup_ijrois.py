#!/usr/bin/env python3
"""
Copies ImageJ ROIs in each recording directory to a backup subdirectory,
with timestamp of backup and when ROIs were created in name of backup files.

Will also backup to a central directory if the AL_IJROI_BACKUP_DIR environment variable
is set.
"""

from argparse import ArgumentParser
from datetime import datetime
from os import getenv
from pathlib import Path
from pprint import pprint
from shutil import copy2
from zipfile import is_zipfile, ZipFile

from ijroi import zip_contents_equal

from hong2p.util import analysis_intermediates_root


def format_time(timestamp: datetime) -> str:
    return timestamp.strftime('%Y-%m-%d_%H%M%S')


def main():
    parser = ArgumentParser()
    # https://stackoverflow.com/questions/6076690
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    assert 0 <= args.verbose <= 2

    verbose = args.verbose > 0
    extra_verbose = args.verbose > 1

    # The filename we are looking for under analysis_root, to be backed up.
    roi_zip_name = 'RoiSet.zip'

    # The sub-directory (within each experiments analysis directory) to be created and
    # used for backups of the RoiSet.zip files.
    backup_dir_name = 'ijroi_backups'

    central_backup_dir_envvar = 'AL_IJROI_BACKUP_DIR'
    central_backup_dir_root = getenv(central_backup_dir_envvar)

    # If True, will delete any backups with equivalent content (minus the timestamps...)
    # before saving them once again. For changing naming convention / etc.
    # Does not apply to central backups.
    _replace_equiv_backups = False

    if _replace_equiv_backups:
        verbose = True

    # if i ever actually use both a fast and non-fast analysis intermediate directory,
    # would need to also check the non-fast one
    analysis_root = analysis_intermediates_root()

    def format_path(path_under_analysis_root):
        """Formats path as relative to analysis_root, for easier reading
        """
        # NOTE: the .relative_to call might fail / behave badly if link points to
        # something not under analysis_root, though I don't have such links now nor plan
        # to make any.
        return str(path_under_analysis_root.relative_to(analysis_root))

    def print_paths(paths_under_analysis_root):
        pprint({format_path(x) for x in paths_under_analysis_root})

    print(f'checking for ImageJ ROIs ({roi_zip_name}) under {analysis_root}')
    if verbose:
        print()

    backup_time = datetime.now()

    if central_backup_dir_root is not None:
        central_backup_dir_root = Path(central_backup_dir_root)
        assert central_backup_dir_root.is_dir()

        central_backup_dir_root = central_backup_dir_root / backup_dir_name
        central_backup_dir_root.mkdir(exist_ok=True)

        if verbose:
            print(f'also backing up ROIs to central {central_backup_dir_root=}\n')

        central_backup_dir = central_backup_dir_root / format_time(backup_time)
        central_backup_dir.mkdir(exist_ok=False)
    else:
        if verbose:
            print('not additionally backing up ROIs to a central directory '
                f'(env var {central_backup_dir_envvar} not set)\n'
            )
        central_backup_dir = None

    roi_zip_glob_str = f'*/*/*/{roi_zip_name}'

    backed_up_this_run = set()
    seen_roi_zips = set()
    # 3 *'s = <date>/<fly_num>/<thorimage_id>
    #
    # I should have always only been saving the RoiSet.zip files under each recording's
    # analysis directory, though there are some symlinks up one level.
    for roi_zip in sorted(analysis_root.glob(roi_zip_glob_str)):

        if verbose:
            print(format_path(roi_zip))

        if roi_zip.is_symlink():
            link_target = roi_zip.resolve()
            assert analysis_root.resolve() in link_target.parents, (
                f'link target {link_target} not under {analysis_root}. '
                'would not have been backed up.'
            )
            if verbose:
                print(f'was symlink (-> {format_path(link_target)}). skipping.\n')

            continue

        # in case there is a symlink in the path, we still know a given path always
        # refers to the same actual file, when considering the set we've seen
        roi_zip = roi_zip.resolve()
        seen_roi_zips.add(roi_zip)

        backup_dir = roi_zip.parent / backup_dir_name
        backup_dir.mkdir(exist_ok=True)

        make_backup = True
        for existing_backup in backup_dir.glob('*.zip'):
            # This checks if ZIP files are equal, but intentially does NOT check
            # timestamps, so that if ROIs are loaded and re-saved (without changing)
            # a new backup does not need to be created.
            if zip_contents_equal(roi_zip, existing_backup):
                if verbose:
                    print(f'already backed up at {format_path(existing_backup)}')

                if _replace_equiv_backups:
                    if verbose:
                        print('deleting existing equivalent backup')

                    existing_backup.unlink()
                    continue

                make_backup = False
                break

        if not make_backup:
            # Since we want all ROIs to be included in each of the timestamp-named
            # directories under central backup root.
            if central_backup_dir is None:

                if verbose:
                    print()
                continue

        with ZipFile(roi_zip, 'r') as zip_file:
            if extra_verbose:
                # Prints three columns: 'File Name', 'Modified', 'Size'
                zip_file.printdir()

            info_list = zip_file.infolist()
            # Each .date_time is a tuple of: (year, month, day, hour, minute, second)
            zip_times = set(x.date_time for x in info_list)

        # If this ever fails, it might mean ImageJ now supports ROI creation time in ZIP
        # file memeber mod time, which could have uses
        assert len(zip_times) == 1

        # Using this, rather than mtime of .zip file, as copying ROIs (without
        # preserving mtime) could give the wrong impression about when the ZIP was saved
        # from ImageJ.
        save_time = datetime(*zip_times.pop())
        save_time_fname = f'{format_time(save_time)}-creation_{roi_zip_name}'

        if make_backup:
            backup = (
                backup_dir / f'{format_time(backup_time)}-backup_{save_time_fname}'
            )
            if verbose:
                print(f'copying to {format_path(backup)}')

            # copy2 preserves modification time (among other things).
            copy2(roi_zip, backup)
            backed_up_this_run.add(roi_zip)

        if central_backup_dir is not None:
            # E.g. '2021-04-28_1_butanal_and_acetone',
            # from 2021-04-28/1/butanal_and_acetone
            experiment_id_prefix = '_'.join(
                roi_zip.relative_to(analysis_root).parent.parts
            )
            # We already have the backup time in the name of central_backup_dir, so no
            # need to also include in filenames here.
            central_backup = (
                central_backup_dir / f'{experiment_id_prefix}_{save_time_fname}'
            )
            if verbose:
                # NOTE: can't use format_path here, as central backup dir not under
                # analysis root
                print(f'copying to central backup at {central_backup}')

            copy2(roi_zip, central_backup)

        if verbose:
            print()


    all_zip_files = set()
    # TODO this also match stuff WITHOUT stuff after the '.zip' part?
    for zip_path in analysis_root.rglob(f'*.zip*'):
        if zip_path.is_symlink():
            continue

        # unlikely to be relevant
        if not is_zipfile(zip_path):
            continue

        # don't want to count backups generated by this script
        if zip_path.parent.name == backup_dir_name:
            continue

        all_zip_files.add(zip_path.resolve())

    assert seen_roi_zips.issubset(all_zip_files)

    zips_not_processed = all_zip_files - seen_roi_zips
    if len(zips_not_processed) > 0:

        wrong_name = {x for x in zips_not_processed if x.name != roi_zip_name}

        # NOTE: if we ever change roi_zip_glob_str (above),
        # this would also need to change.
        wrong_depth = {x for x in zips_not_processed
            if Path(*x.parts[:-4]) != analysis_root
        }

        assert wrong_name | wrong_depth == zips_not_processed, \
            'some zips not processed for unexpected reasons'

        if extra_verbose:
            print(f'ZIP files (under {analysis_root}) that were not backed up:')

            if len(wrong_name) > 0:
                print(f'...because their filename was not {roi_zip_name}:')
                print_paths(wrong_name)
                print()

            if len(wrong_depth) > 0:
                # should also change if roi_zip_glob_str ever does
                print(f'...because they were not 3 directories below analysis root:')
                print_paths(wrong_depth)
                print()

    if len(backed_up_this_run) == 0:
        print('nothing new to backup')
    else:
        print('newly backed up:')
        print_paths(backed_up_this_run)


if __name__ == '__main__':
    main()

