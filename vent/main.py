#!/usr/bin/env python
import argparse
import sys
import os
import time
from vent import prefs
import vent.io as io

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

def set_valves_save_position():
    print("Terminating program; closing vents...")
    if not args.simulation:
        time.sleep(0.01)
        HAL = io.Hal( config_file = 'vent/io/config/devices.ini')
        for i in range(10):
            HAL.setpoint_in = 0
            HAL.setpoint_ex = 1 
            time.sleep(0.01)
    else:
        print("Terminating simulation.")

def main():
    args = parse_cmd_args()
    try:
        coordinator = get_coordinator(single_process=args.single_process, sim_mode=args.simulation)
        app, gui = launch_gui(coordinator)
        sys.exit(app.exec_())
    finally:
        set_valves_to_save_position()
        print("...done")
        
    # TODO: gui.main(ui_control_module)

    # TODO: use signal for mor flexible termination, e.g.
    # signal.signal(signal.SIGINT, set_valves_to_save_position)   # Keyboard interrupt
    # signal.signal(signal.SIGTERM, set_valves_to_save_position)  # Termination signal


if __name__ == '__main__':
    main()
