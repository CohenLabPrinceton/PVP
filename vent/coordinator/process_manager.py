import threading
import time
import os
from vent.coordinator.ipc import IPC
import vent
from time import sleep
from vent.coordinator.ipc import IPCCommand, IPCMessage


class ProcessManager:
    # Functions:
    def __init__(self, sim_mode, startCommandLine=None, maxHeartbeatInterval=None):
        self.command_line = None  # TODO: what is this?
        self.max_heartbeat_interval = None
        self.previous_timestamp = None
        self.controller_process_id = os.fork()
        if self.controller_process_id == 0:
            self.local_coordinator = vent.coordinator.coordinator.get_coordinator(sim_mode=sim_mode, single_process=True)
            self.ipc = IPC(listen=False)
            self.receive_thread = threading.Thread(target=self.__child_receive_loop, daemon=True)
            self.send_thread = threading.Thread(target=self.__child_send_loop, daemon=True)
            self.receive_thread.start()
            self.send_thread.start()
            while True:
                sleep(1)
        else:
            self.parent_thread = threading.Thread(target=self.__parent_loop, daemon=True)
            self.parent_thread.start()

    def startProcess(self):
        pass

    def stopProcess(self):
        pass

    def restartProcess(self):
        pass

    def heartbeat(self, timestamp):
        # if no heartbeat in maxInterval restart
        pass

    def __parent_loop(self):
        while True:
            # TODO: monitor the process, restart if process hang
            print('Parent process')
            time.sleep(10)

    def __child_receive_loop(self):
        while True:
            # TODO: run the controller loop
            msg = self.ipc.recv_msg()
            print(f'child process received: {msg.command}')
            if msg.command == IPCCommand.START:
                self.local_coordinator.start()
            elif msg.command == IPCCommand.STOP:
                self.local_coordinator.stop()
            elif msg.command == IPCCommand.GET_CONTROL:
                control_setting = self.local_coordinator.get_control(msg.args)
                self.ipc.send_msg(IPCMessage(IPCCommand.GET_CONTROL, control_setting))
            elif msg.command == IPCCommand.SET_CONTROL:
                self.local_coordinator.set_control(msg.args)
            else:
                raise NotImplementedError(f'Error: {msg.command} not implemented')

    def __child_send_loop(self):
        while True:
            print(f'child process sending')
            sensor_values = self.local_coordinator.get_sensors()
            msg = IPCMessage(IPCCommand.GET_SENSORS, sensor_values)
            print(f'child process sending: {msg.command}')
            self.ipc.send_msg(msg)
            # sleep 10 ms
            time.sleep(0.01)

# Instance Variables:
#     commandLine
#     maxHeartbeatInterval
#     previousTimestamp
#     processId
