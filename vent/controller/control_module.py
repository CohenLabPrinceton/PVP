import time
from typing import List
import threading
import numpy as np
import copy
from collections import deque

from vent.common.message import SensorValues, ControlSetting, Alarm, AlarmSeverity, ControlSettingName



class ControlModuleBase:
    """This is an abstract class for controlling simulation and hardware.

    1. All internal variables fall in three classes, denoted by the beginning of the variable:
        - "COPY_varname": These are copies (see 1.) that are regularly sync'ed with internal variables.
        - "__varname":    These are variables only used in the ControlModuleBase-Class
        - "_varname":     These are variables used in derived classes.

    2. Internal variables should only to be accessed though the set_ and get_ functions.
        These functions act on COPIES of internal variables ("__" and "_"), that are sync'd every few
        iterations. How often this is done is adjusted by the variable
        self._NUMBER_CONTROLL_LOOPS_UNTIL_UPDATE. To avoid multiple threads manipulating the same 
        variables at the same time, every manipulation of "COPY_" is surrounded by a thread lock.

    Public Methods:
        get_sensors():                     Returns a copy of the current sensor values.
        get_alarms():                      Returns a List of all alarms, active and logged
        get_active_alarms():               Returns a Dictionary of all currently active alarms.
        get_logged_alarms():               Returns a List of logged alarms, up to maximum lengh of self._RINGBUFFER_SIZE
        get_control(ControlSetting):       Sets a controll-setting. Is updated at latest within self._NUMBER_CONTROLL_LOOPS_UNTIL_UPDATE
        get_past_waveforms():              Returns a List of waveforms of pressure and volume during at the last N breath cycles, N<self._RINGBUFFER_SIZE, AND clears this archive.
        start():                           Starts the main-loop of the controller
        stop():                            Stops the main-loop of the controller
    """

    def __init__(self):

        #####################  Algorithm/Program parameters  ##################
        # Hyper-Parameters
        self._LOOP_UPDATE_TIME                   = 0.01    # Run the main control loop every 0.01 sec
        self._NUMBER_CONTROLL_LOOPS_UNTIL_UPDATE = 10      # After every 10 main control loop iterations, update COPYs.
        self._RINGBUFFER_SIZE                    = 100     # Maximum number of breath cycles kept in memory

        #########################  Control management  #########################

        # This is what the machine has controll over
        self._control_signal = 0              # State of a valve on the inspiratory side

        # Internal Control variables. "SET" indicates that this is set.
        self.__SET_PIP = 22         # Target PIP pressure
        self.__SET_PIP_TIME = 0.8   # Target time to reach PIP in seconds
        self.__SET_PEEP = 5         # Target PEEP pressure
        self.__SET_PEEP_TIME = 0.5  # Target time to reach PEEP from PIP plateau
        self.__SET_BPM = 15         # Target breaths per minute
        self.__SET_I_PHASE = 1.3    # Target duration of inspiratory phase

        # Derived internal control variables - fully defined by numbers above
        self.__SET_CYCLE_DURATION = 60 / self.__SET_BPM
        self.__SET_E_PHASE        = self.__SET_CYCLE_DURATION - self.__SET_I_PHASE
        self.__SET_T_PLATEAU      = self.__SET_I_PHASE - self.__SET_PIP_TIME
        self.__SET_T_PEEP         = self.__SET_E_PHASE - self.__SET_PEEP_TIME

        #########################  Data management  #########################

        # These are measurements from the last breath cycle.
        self._DATA_PIP = None         # Measured value of PIP
        self._DATA_PIP_TIME = None    # Measured time of reaching PIP plateau
        self._DATA_PEEP = None        # Measured valued of PEEP
        self._DATA_I_PHASE = None     # Measured duration of inspiratory phase
        self.__DATA_FIRST_PEEP = None # Time when PEEP is reached first
        self.__DATA_LAST_PEEP = None  # Last time of PEEP - by definition end of breath cycle
        self._DATA_BPM = None         # Measured breathing rate, by definition 60sec / length_of_breath_cycle
        self._DATA_VTE = None         # Maximum air displacement in last breath cycle
        self._DATA_P = 0              # Last measurements of the proportional term for PID-control
        self._DATA_I = 0              # Last measurements of the integral term for PID-control
        self._DATA_D = 0              # Last measurements of the differential term for PID-control

        # Parameters to keep track of breath-cycle
        self.__cycle_start = time.time()
        self.__cycle_waveform = None                             # To build up the current cycle's waveform
        self.__cycle_waveform_archive = deque(maxlen = self._RINGBUFFER_SIZE)          # An archive of past waveforms.

        # These are measurements that change from timepoint to timepoint
        self._DATA_PRESSURE = 0
        self.__DATA_VOLUME = 0
        self._DATA_Qin = 0            # Measurement of the airflow in
        self._DATA_Qout = 0           # Measurement of the airflow out
        self._DATA_dpdt = 0           # Current sample of the rate of change of pressure dP/dt in cmH2O/sec
        self._last_update = time.time()


        #########################  Alarm management  #########################
        self.__active_alarms = {}     # Dictionary of active alarms
        self.__logged_alarms = deque(maxlen = self._RINGBUFFER_SIZE)     # List of all resolved alarms

        # Variable limits to raise alarms, initialized as small deviation of what the controller initializes
        self.__PIP_min = self.__SET_PIP - 0.2
        self.__PIP_max = self.__SET_PIP + 0.2
        self.__PIP_lastset = time.time()
        self.__PIP_time_min = self.__SET_PIP_TIME - 0.5
        self.__PIP_time_max = self.__SET_PIP_TIME + 0.5
        self.__PIP_time_lastset = time.time()
        self.__PEEP_min = self.__SET_PEEP * 0.9
        self.__PEEP_max = self.__SET_PEEP * 1.1
        self.__PEEP_lastset = time.time()
        self.__bpm_min = self.__SET_BPM * 0.9
        self.__bpm_max = self.__SET_BPM * 1.1
        self.__bpm_lastset = time.time()
        self.__I_phase_min = self.__SET_I_PHASE * 0.9
        self.__I_phase_max = self.__SET_I_PHASE * 1.1
        self.__I_phase_lastset = time.time()

        ############### Initialize COPY variables for threads  ##############
        # COPY variables that later updated on a regular basis
        self.COPY_active_alarms = {}
        self.COPY_logged_alarms = list(self.__logged_alarms)
        self.COPY_sensor_values = SensorValues()

        ###########################  Threading init  #########################
        # Run the start() method as a thread
        self._loop_counter = 0
        self._running = False
        self._lock = threading.Lock()
        self._alarm_to_COPY()  #These require the lock
        self._initialize_set_to_COPY()

        self.__thread = threading.Thread(target=self._start_mainloop, daemon=True)
        self.__thread.start()


    def _initialize_set_to_COPY(self):
        self._lock.acquire()
        # Copy of the SET variables for threading.
        self.COPY_SET_PIP       = self.__SET_PIP 
        self.COPY_SET_PIP_TIME  = self.__SET_PIP_TIME
        self.COPY_SET_PEEP      = self.__SET_PEEP
        self.COPY_SET_PEEP_TIME = self.__SET_PEEP_TIME
        self.COPY_SET_BPM       = self.__SET_BPM
        self.COPY_SET_I_PHASE   = self.__SET_I_PHASE
        self._lock.release()

    def _alarm_to_COPY(self):
        self._lock.acquire()
        # Update the alarms
        self.COPY_active_alarms = self.__active_alarms.copy()
        self.COPY_logged_alarms = list(self.__logged_alarms)

        # The alarm thresholds
        self.COPY_PIP_min = self.__PIP_min
        self.COPY_PIP_max = self.__PIP_max
        self.COPY_PIP_lastset = self.__PIP_lastset
        self.COPY_PIP_time_min = self.__PIP_time_min 
        self.COPY_PIP_time_max = self.__PIP_time_max
        self.COPY_PIP_time_lastset = self.__PIP_time_lastset
        self.COPY_PEEP_min = self.__PEEP_min
        self.COPY_PEEP_max = self.__PEEP_max
        self.COPY_PEEP_lastset = self.__PEEP_lastset
        self.COPY_bpm_min = self.__bpm_min 
        self.COPY_bpm_max = self.__bpm_max
        self.COPY_bpm_lastset = self.__bpm_lastset
        self.COPY_I_phase_min = self.__I_phase_min
        self.COPY_I_phase_max = self.__I_phase_max
        self.COPY_I_phase_lastset = self.__I_phase_lastset

        self._lock.release()

    def _sensor_to_COPY(self):
        # These variables have to come from the hardware
        self._lock.acquire()
        # Make sure you have acquire and release!
        self._lock.release()
        pass

    def _controls_from_COPY(self):
        # Update SET variables
        self._lock.acquire()

        #Update values
        self.__SET_PIP       = self.COPY_SET_PIP
        self.__SET_PIP_TIME  = self.COPY_SET_PIP_TIME
        self.__SET_PEEP      = self.COPY_SET_PEEP
        self.__SET_PEEP_TIME = self.COPY_SET_PEEP_TIME
        self.__SET_BPM       = self.COPY_SET_BPM
        self.__SET_I_PHASE   = self.COPY_SET_I_PHASE

        #Update derived values
        self.__SET_CYCLE_DURATION = 60 / self.__SET_BPM
        self.__SET_E_PHASE = self.__SET_CYCLE_DURATION - self.__SET_I_PHASE
        self.__SET_T_PLATEAU = self.__SET_I_PHASE - self.__SET_PIP_TIME
        self.__SET_T_PEEP = self.__SET_E_PHASE - self.__SET_PEEP_TIME

        #Update new confidence intervals
        self.__PIP_min          = self.COPY_PIP_min
        self.__PIP_max          = self.COPY_PIP_max 
        self.__PIP_lastset      = self.COPY_PIP_lastset  
        self.__PIP_time_min     = self.COPY_PIP_time_min  
        self.__PIP_time_max     = self.COPY_PIP_time_max  
        self.__PIP_time_lastset = self.COPY_PIP_time_lastset  
        self.__PEEP_min         = self.COPY_PEEP_min 
        self.__PEEP_max         = self.COPY_PEEP_max  
        self.__PEEP_lastset     = self.COPY_PEEP_lastset  
        self.__bpm_min          = self.COPY_bpm_min 
        self.__bpm_max          = self.COPY_bpm_max  
        self.__bpm_lastset      = self.COPY_bpm_lastset  
        self.__I_phase_min      = self.COPY_I_phase_min  
        self.__I_phase_max      = self.COPY_I_phase_max 
        self.__I_phase_lastset  = self.COPY_I_phase_lastset  

        self._lock.release()

    def __test_critical_levels(self, min, max, value, name):
        '''
        This tests whether a variable is within bounds.
        If it is, and an alarm existed, then the "alarm_end_time" is set.
        If it is NOT, a new alarm is generated and appendede to the alarm-list.
        Input:
            min:           minimum value  (e.g. 2)
            max:           maximum value  (e.g. 5)
            value:         test value   (e.g. 3)
            name:          parameter type (e.g. "PIP", "PEEP" etc.)
        '''
        if (value < min) or (value > max):  # If the variable is not within limits
            if name not in self.__active_alarms.keys():  # And and alarm for that variable doesn't exist yet -> RAISE ALARM.
                new_alarm = Alarm(alarm_name=name, is_active=True, severity=AlarmSeverity.RED, \
                                  alarm_start_time=time.time(), alarm_end_time=None)
                self.__active_alarms[name] = new_alarm
        else:  # Else: if the variable is within bounds,
            if name in self.__active_alarms.keys():  # And an alarm exists -> inactivate it.
                old_alarm = self.__active_alarms[name]
                old_alarm.alarm_end_time = time.time()
                old_alarm.is_active = False
                self.__logged_alarms.append(old_alarm)
                del self.__active_alarms[name]

    def __analyze_last_waveform(self):
        ''' This goes through the last waveform, and updates VTE, PEEP, PIP, PIP_TIME, I_PHASE, FIRST_PEEP and BPM.'''
        data = self.__cycle_waveform_archive[-1]
        if data is not None:  # Only if there was a previous cycle
            phase = data[:, 0]
            pressure = data[:, 1]
            volume = data[:, 2]

            self._DATA_VTE = np.max(volume) - np.min(volume)

            # get the pressure niveau heuristically (much faster than fitting)
            # 20/80 percentile of pressure values below/above mean
            # Assumption: waveform is mostly between both plateaus
            self._DATA_PEEP = np.percentile(pressure[ pressure < np.mean(pressure)], 20 )
            self._DATA_PIP  = np.percentile(pressure[ pressure > np.mean(pressure)], 80 )

            # measure time of reaching PIP, and leaving PIP
            self._DATA_PIP_TIME = phase[np.min(np.where(pressure > self._DATA_PIP))]
            self._DATA_I_PHASE = phase[np.max(np.where(pressure > self._DATA_PIP))]

            # and measure the same for PEEP
            self.__DATA_FIRST_PEEP = phase[np.min(np.where(np.logical_and(pressure < self._DATA_PEEP, phase > 1)))]
            self._DATA_BPM = 60. / phase[-1]  # 60 sec divided by the duration of last waveform

    def __update_alarms(self):
        ''' This goes through the values obtained from the last waveform, and updates alarms.'''
        if self.__cycle_waveform_archive[-1] is not None:
            self.__test_critical_levels(min=self.__PIP_min, max=self.__PIP_max, value=self._DATA_PIP, name="PIP")
            self.__test_critical_levels(min=self.__PIP_time_min, max=self.__PIP_time_max, value=self._DATA_PIP_TIME, name="PIP_TIME")
            self.__test_critical_levels(min=self.__PEEP_min, max=self.__PEEP_max, value=self._DATA_PEEP, name="PEEP")
            self.__test_critical_levels(min=self.__bpm_min, max=self.__bpm_max, value=self._DATA_BPM, name="BREATHS_PER_MINUTE")
            self.__test_critical_levels(min=self.__I_phase_min, max=self.__I_phase_max, value=self._DATA_I_PHASE, name="I_PHASE")

    def get_sensors(self) -> SensorValues:
        # Make sure to return a copy of the instance
        self._lock.acquire()
        cp = copy.deepcopy( self.COPY_sensor_values )
        self._lock.release()
        return cp

    def get_alarms(self) -> List[Alarm]:
        # Returns all alarms as a list
        self._lock.acquire()
        new_alarm_list = self.COPY_logged_alarms.copy()
        for alarm_key in self.COPY_active_alarms.keys():
            new_alarm_list.append(self.COPY_active_alarms[alarm_key])
        self._lock.release()
        return new_alarm_list

    def get_active_alarms(self):
        # Returns only the active alarms
        self._lock.acquire()
        active_alarms = self.COPY_active_alarms.copy() # Make sure to return a copy
        self._lock.release()
        return active_alarms

    def get_logged_alarms(self) -> List[Alarm]:
        # Returns only the inactive alarms
        self._lock.acquire()
        logged_alarms = self.COPY_logged_alarms.copy()
        self._lock.release()
        return logged_alarms

    def set_control(self, control_setting: ControlSetting):
        ''' Updates the entry of COPY contained in the control settings'''
        self._lock.acquire()

        if control_setting.name == ControlSettingName.PIP:
            self.COPY_SET_PIP = control_setting.value
            self.COPY_PIP_min = control_setting.min_value
            self.COPY_PIP_max = control_setting.max_value
            self.COPY_PIP_lastset = control_setting.timestamp

        elif control_setting.name == ControlSettingName.PIP_TIME:
            self.COPY_SET_PIP_TIME = control_setting.value
            self.COPY_PIP_time_min = control_setting.min_value
            self.COPY_PIP_time_max = control_setting.max_value
            self.COPY_PIP_time_lastset = control_setting.timestamp

        elif control_setting.name == ControlSettingName.PEEP:
            self.COPY_SET_PEEP = control_setting.value
            self.COPY_PEEP_min = control_setting.min_value
            self.COPY_PEEP_max = control_setting.max_value
            self.COPY_PEEP_lastset = control_setting.timestamp

        elif control_setting.name == ControlSettingName.BREATHS_PER_MINUTE:
            self.COPY_SET_BPM = control_setting.value
            self.COPY_bpm_min = control_setting.min_value
            self.COPY_bpm_max = control_setting.max_value
            self.COPY_bpm_lastset = control_setting.timestamp

        elif control_setting.name == ControlSettingName.INSPIRATION_TIME_SEC:
            self.COPY_SET_I_PHASE = control_setting.value
            self.COPY_I_phase_min = control_setting.min_value
            self.COPY_I_phase_max = control_setting.max_value
            self.COPY_I_phase_lastset = control_setting.timestamp

        else:
            raise KeyError("You cannot set the variabe: " + str(control_setting.name))

        self._lock.release()

    def get_control(self, control_setting_name: ControlSettingName) -> ControlSetting:
        ''' Gets values of the COPY of the control settings. '''
        
        self._lock.acquire()
        if control_setting_name == ControlSettingName.PIP:
            return_value = ControlSetting(control_setting_name,
                                  self.COPY_SET_PIP,
                                  self.COPY_PIP_min,
                                  self.COPY_PIP_max,
                                  self.COPY_PIP_lastset)
        elif control_setting_name == ControlSettingName.PIP_TIME:
            return_value = ControlSetting(control_setting_name,
                                  self.COPY_SET_PIP_TIME,
                                  self.COPY_PIP_time_min,
                                  self.COPY_PIP_time_max,
                                  self.COPY_PIP_time_lastset, )
        elif control_setting_name == ControlSettingName.PEEP:
            return_value = ControlSetting(control_setting_name,
                                  self.COPY_SET_PEEP,
                                  self.COPY_PEEP_min,
                                  self.COPY_PEEP_max,
                                  self.COPY_PEEP_lastset)
        elif control_setting_name == ControlSettingName.BREATHS_PER_MINUTE:
            return_value = ControlSetting(control_setting_name,
                                  self.COPY_SET_BPM,
                                  self.COPY_bpm_min,
                                  self.COPY_bpm_max,
                                  self.COPY_bpm_lastset)
        elif control_setting_name == ControlSettingName.INSPIRATION_TIME_SEC:
            return_value = ControlSetting(control_setting_name,
                                  self.COPY_SET_I_PHASE,
                                  self.COPY_I_phase_min,
                                  self.COPY_I_phase_max,
                                  self.COPY_I_phase_lastset)
        else:
            raise KeyError("You cannot set the variabe: " + str(control_setting_name))

        self._lock.release()

        return return_value


    def __get_PID_error(self, ytarget, yis, dt):
        error_new = ytarget - yis                   # New value of the error

        RC = 0.5  # Time constant in seconds
        s = dt / (dt + RC)
        self._DATA_I = self._DATA_I + s*(error_new - self._DATA_I)     # Integral term on some timescale RC
        self._DATA_D = error_new - self._DATA_P
        self._DATA_P = error_new

    def __calculate_control_signal(self):
        self.control_signal  = 160            # Some setting for the maximum flow.
        self.control_signal -= 40*self._DATA_P
        self.control_signal -=  0*self._DATA_I
        self.control_signal -=  0*self._DATA_D

    def _get_control_signal(self):
        return self.control_signal

    def __get_pressure_derivative(self, dt):
        if self.__cycle_waveform is None:
            sample_dPdt = 0
        else:
            latest_sample = self.__cycle_waveform[-1]  #Format of the last sample: [time|pressure|volume]
            sample_dPdt = (self._DATA_PRESSURE - latest_sample[1])/dt     # this is dP/dt at one moment in time

        s = dt / (0.02 + dt)  #send this through 20ms LP filter
        self._DATA_dpdt = self._DATA_dpdt + s * (sample_dPdt - self._DATA_dpdt)


    def _PID_update(self, dt):
        ''' 
        This instantiates the control algorithms.
        During the breathing cycle, it goes through the four states:
           1) Rise to PIP
           2) Sustain PIP pressure
           3) Quick fall to PEEP
           4) Sustaint PEEP pressure
        Once the cycle is complete, it checks the cycle for any alarms, and starts a new one.
        A record of pressure/volume waveforms is kept in self.__cycle_waveform_archive

            dt: Time since last update in seconds 

        '''
        now = time.time()
        cycle_phase = now - self.__cycle_start
        next_cycle = False

        self.__DATA_VOLUME += dt * ( self._DATA_Qin - self._DATA_Qout )  # Integrate what has happened within the last few seconds from the measurements of Qin and Qout

        if cycle_phase < self.__SET_PIP_TIME:
            self.__get_pressure_derivative(dt = dt)
            target_slope = (self.__SET_PIP - self.__SET_PEEP) / self.__SET_PIP_TIME
            self.__get_PID_error(yis = self._DATA_dpdt, ytarget = target_slope, dt = dt)   # here, we control dP/dt   
            self.__calculate_control_signal()

            # STATE CONTROL: to PIP, air in as fast as possible
            # self._Qin = 1
            # self._Qout = 0
            # if self._DATA_PRESSURE > self.__SET_PIP:
            #     self._Qin = 0
        elif cycle_phase < self.__SET_I_PHASE:                          # then, we control PIP 
            self.__get_PID_error(yis = self._DATA_PRESSURE, ytarget = self.__SET_PIP, dt = dt)
            self.__calculate_control_signal()

            # STATE CONTROL: keep PIP plateau, let air in if below
            # self._Qin = 0
            # self._Qout = 0
            # if self._DATA_PRESSURE < self.__SET_PIP:
            #     self._Qin = 1
        elif cycle_phase < self.__SET_PEEP_TIME + self.__SET_I_PHASE:  # then, we drop pressure as fast as possible
            self.control_signal = np.inf
            if self._DATA_PRESSURE < self.__SET_PEEP:
                self.control_signal = 0

            # self._Qin = 0
            # self._Qout = 1
            # if self._DATA_PRESSURE < self.__SET_PEEP:
            #     self._Qout = 0
        elif cycle_phase < self.__SET_CYCLE_DURATION:                   # and control around PEEP
            self.__get_PID_error(yis = self._DATA_PRESSURE, ytarget = self.__SET_PEEP, dt = dt)
            self.__calculate_control_signal()

            # STATE CONTROL: keeping PEEP, let air in if below
            # self._Qin = 0
            # self._Qout = 0
            # if self._DATA_PRESSURE < self.__SET_PEEP:
            #     self._Qin = 1
        else:
            self.__cycle_start = time.time()  # New cycle starts
            self.__DATA_VOLUME = 0            # ... start at zero volume in the lung
            self._DATA_dpdt    = 0            # and restart the rolling average for the dP/dt estimation
            next_cycle = True
        
        if next_cycle or (self.__cycle_waveform is None):  # if a new breath cycle has started, or program just started
            self.__cycle_waveform_archive.append( self.__cycle_waveform )
            self.__cycle_waveform = np.array([[0, self._DATA_PRESSURE, self.__DATA_VOLUME]])
            self.__analyze_last_waveform()    # Analyze last waveform
            self.__update_alarms()            # Run alarm detection over last cycle's waveform
            self._sensor_to_COPY()            # Get the fit values from the last waveform directly into sensor values
        else:
            self.__cycle_waveform = np.append(self.__cycle_waveform, [[cycle_phase, self._DATA_PRESSURE, self.__DATA_VOLUME]], axis=0)

    def get_past_waveforms(self):
        # Returns a list of past waveforms.
        # Format:
        #     Returns a list of [Nx3] waveforms, of [time, pressure, volume]
        #     Most recent entry is waveform_list[-1]
        # Note:
        #     After calling this function, archive is emptied!
        self._lock.acquire()
        archive = list( self.__cycle_waveform_archive ) # Make sure to return a copy as a list
        self.__cycle_waveform_archive = deque(maxlen = self._RINGBUFFER_SIZE)
        self.__cycle_waveform_archive.append(archive[-1])
        self._lock.release()
        return archive

    def _start_mainloop(self):
        # This will depend on simulation or reality
        pass   

    def start(self):
        if not self.__thread.is_alive():  # If the previous thread has been stopped, make a new one.
            self._running = True
            self.__thread = threading.Thread(target=self._start_mainloop, daemon=True)
            self.__thread.start()
        else:
            print("Main Loop already running.")

    def stop(self):
        if self.__thread.is_alive():
            self._running = False
        else:
            print("Main Loop is not running.")



class ControlModuleDevice(ControlModuleBase):
    # Implement ControlModuleBase functions
    pass




class Balloon_Simulator:
    '''
    This is a imple physics simulator for inflating a balloon. 
    For math, see https://en.wikipedia.org/wiki/Two-balloon_experiment
    '''

    def __init__(self, leak, delay):
        # Hard parameters for the simulation
        self.max_volume = 6  # Liters  - 6?
        self.min_volume = 1.5  # Liters - baloon starts slightly inflated.
        self.PC = 20  # Proportionality constant that relates pressure to cm-H2O
        self.P0 = 0  # Minimum pressure.
        self.leak = leak
        self.delay = delay

        self.temperature = 37  # keep track of this, as is important output variable
        self.humidity = 90
        self.fio2 = 60

        # Dynamical parameters - these are the initial conditions
        self.current_flow = 0  # in unit  liters/sec
        self.current_pressure = 0  # in unit  cm-H2O
        self.r_real = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)  # size of the lung
        self.current_volume = self.min_volume  # in unit  liters

    def get_pressure(self):
        return self.current_pressure

    def get_volume(self):
        return self.current_volume

    def set_flow(self, Qin, Qout):
        self.current_flow = Qin - Qout

    def update(self, dt):  # Performs an update of duration dt [seconds]
        self.current_volume += self.current_flow * dt

        if self.leak:
            RC = 5  # pulled 5 sec out of my hat
            s = dt / (RC + dt)
            self.current_volume = self.current_volume + s * (self.min_volume - self.current_volume)

        # This is fromt the baloon equation, uses helper variable (the baloon radius)
        r_target = (3 * self.current_volume / (4 * np.pi)) ** (1 / 3)
        r0 = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)

        # Delay -> Expansion takes time
        if self.delay:
            RC = 0.1  # pulled these 100ms out of my hat
            s = dt / (RC + dt)
            self.r_real = self.r_real + s * (r_target - self.r_real)
        else:
            self.r_real = r_target

        self.current_pressure = self.P0 + (self.PC / (r0 ** 2 * self.r_real)) * (1 - (r0 / self.r_real) ** 6)

        # Temperature, humidity and o2 fluctuations modelled as OUprocess
        self.temperature = self.OUupdate(self.temperature, dt=dt, mu=37, sigma=0.3, tau=1)
        self.fio2 = self.OUupdate(self.fio2, dt=dt, mu=60, sigma=5, tau=1)
        self.humidity = self.OUupdate(self.humidity, dt=dt, mu=90, sigma=5, tau=1)
        if self.humidity > 100:
            self.humidity = 100

    def OUupdate(self, variable, dt, mu, sigma, tau):
        '''
        This is a simple function to produce an OU process.
        It is used as model for fluctuations in measurement variables.
        inputs:
        variable:   float     value at previous time step
        dt      :   timestep
        mu      :   mean
        sigma   :   noise amplitude
        tau     :   time scale
        returns:
        new_variable :  value of "variable" at next time step
        '''
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)
        new_variable = variable + dt * (-(variable - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()
        return new_variable

class ControlModuleSimulator(ControlModuleBase):
    # Implement ControlModuleBase functions
    def __init__(self):
        ControlModuleBase.__init__(self)
        self.Balloon = Balloon_Simulator(leak=True, delay=False)          # This is the simulation
        self._sensor_to_COPY()

        self._DATA_Qin    =    0.6  # Fixed flow of 0.6 l/min
        self._DATA_Qout   =    0    # Later set by control

    def __SimulatedPropValve(self, x, dt):
        '''
        This simulates the action of a proportional valve.
        Flow-current-curve eye-balled from the datasheet of SMC PVQ31-5G-23-01N  
        https://www.ocpneumatics.com/content/pdfs/PVQ.pdf
        
        x:  Input current [mA]
        dt: Time since last setting in seconds [for the LP filter]
        '''
        RC = 0.05 # 50 ms delay
        flow_new = 1.0*(np.tanh(0.03*(x - 130)) + 1)
        if x>160:
            flow_new = 1.72  #Maximum, ~100 l/min
        if x<0:
            flow_new = 0

        s = dt / (RC + dt)
        self._DATA_Qout = self._DATA_Qout + s * (flow_new - self._DATA_Qout)

    def _sensor_to_COPY(self):
        # And the sensor measurements
        self._lock.acquire()
        self.COPY_sensor_values = SensorValues(pip=self._DATA_PIP,
                                          peep=self._DATA_PEEP,
                                          fio2=self.Balloon.fio2,
                                          temp=self.Balloon.temperature,
                                          humidity= self.Balloon.humidity,
                                          pressure=self.Balloon.current_pressure,
                                          vte=self._DATA_VTE,
                                          breaths_per_minute=self._DATA_BPM,
                                          inspiration_time_sec=self._DATA_I_PHASE,
                                          timestamp=time.time(),
                                          loop_counter = self._loop_counter)
        self._lock.release()

    def _start_mainloop(self):
        # start running, this should be run as a thread! 
        # Compare to initialization in Base Class!

        update_copies = self._NUMBER_CONTROLL_LOOPS_UNTIL_UPDATE

        while self._running:
            time.sleep(self._LOOP_UPDATE_TIME)
            self._loop_counter += 1
            now = time.time()

            # Only three sensors "are connected". One to the pressure in the balloon, and two to flow-in and flow-out
            self.Balloon.update(dt = now - self._last_update)
            self._DATA_PRESSURE = self.Balloon.get_pressure()

            self._PID_update(dt = now - self._last_update)  # Updates the PID Controller

            x = self._get_control_signal()
            self.__SimulatedPropValve(x, dt = now - self._last_update)
            self.Balloon.set_flow(self._DATA_Qin, self._DATA_Qout)

            self._last_update = now

            if update_copies == 0:
                self._controls_from_COPY()     # Update controls from possibly updated values as a chunk
                self._alarm_to_COPY()          # Copy current alarms and settings to COPY
                self._sensor_to_COPY()         # Copy sensor values to COPY
                update_copies = self._NUMBER_CONTROLL_LOOPS_UNTIL_UPDATE
            else:
                update_copies -= 1

    def heartbeat(self):
        '''only used for fiddling'''
        return self._loop_counter




def get_control_module(sim_mode=False):
    if sim_mode == True:
        return ControlModuleSimulator()
    else:
        return ControlModuleDevice()