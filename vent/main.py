#!/usr/bin/env python
import argparse
import sys
import os
from vent import prefs
from vent.gui.main import launch_gui
from vent.coordinator.coordinator import get_coordinator


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    # TODO: maybe we should add a mode without UI display, so this would only have command line interface?
    parser.add_argument('--simulation',
                        help='run as simulation using virtual sensors and actuators (default: False)',
                        action='store_true')
    parser.add_argument('--single_process',
                        help='running UI and coordinator within one process (default: False)',
                        action='store_true')
    return parser.parse_args()

def make_vent_dirs():
    """
    Make a directory to store logs, data, and user configuration in ``<user director>/vent``

    Creates::

        ~/vent
        ~/vent/logs - for storage of event and alarm logs
        ~/vent/data - for storage of waveform data
    """

    # root, log, and data vent directories
    vent_dir = os.path.join(os.path.expanduser('~'), 'vent')
    log_dir = os.path.join(vent_dir, 'logs')
    data_dir = os.path.join(vent_dir, 'data')

    # create directories if they don't exist already
    for make_dir in (vent_dir, log_dir, data_dir):
        if not os.path.exists(make_dir):
            os.mkdir(make_dir)

    # store them as config values
    prefs.set_pref('VENT_DIR', vent_dir)
    prefs.set_pref('LOG_DIR', log_dir)
    prefs.set_pref('DATA_DIR', data_dir)



def main():
    make_vent_dirs()
    args = parse_cmd_args()
    coordinator = get_coordinator(single_process=args.single_process, sim_mode=args.simulation)
    app, gui = launch_gui(coordinator)
    sys.exit(app.exec_())


    # TODO: gui.main(ui_control_module)


if __name__ == '__main__':
    main()
