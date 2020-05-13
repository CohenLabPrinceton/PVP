from abc import ABC, abstractmethod
from vent.io.devices.pins import Pin, PWMOutput

import numpy as np


class SolenoidBase(ABC):
    """ An abstract baseclass that defines methods using valve terminology.
    Also allows configuring both normally _open and normally closed valves (called the "form" of the valve).
    """
    _FORMS = {'Normally Closed': 0,
              'Normally Open': 1}

    def __init__(self, form='Normally Closed'):
        """

        Args:
            form (str): The form of the solenoid; can be either `Normally Open` or `Normally Closed`
        """
        self.form = form

    @property
    def form(self) -> str:
        """ Returns the human-readable form of the valve."""
        return dict(map(reversed, self._FORMS.items()))[self._form]

    @form.setter
    def form(self, form):
        """ Performs validation on requested form and then sets it.

        Args:
            f (str): The form of the solenoid; can be either `Normally Open` or `Normally Closed`
        """
        if form not in self._FORMS.keys():
            raise ValueError('form must be either NC for Normally Closed or NO for Normally Open')
        else:
            self._form = self._FORMS[form]

    @abstractmethod
    def open(self):
        """ Energizes valve if Normally Closed. De-energizes if Normally Open."""

    @abstractmethod
    def close(self):
        """ De-energizes valve if Normally Closed. Energizes if Normally Open."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """ Returns True if valve is open, False if it is closed"""
        pass


class OnOffValve(SolenoidBase, Pin):
    """ An extension of vent.io.iobase.Pin which uses valve terminology for its methods.
    Also allows configuring both normally _open and normally closed valves (called the "form" of the valve).
    """
    _FORMS = {'Normally Closed': 0,
              'Normally Open': 1}

    def __init__(self, pin, form='Normally Closed', pig=None):
        """

        Args:
            pin (int): The number of the pin to use
            form (str): The form of the solenoid; can be either `Normally Open` or `Normally Closed`
            pig (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        self.form = form
        Pin.__init__(self, pin, pig)
        SolenoidBase.__init__(self, form=form)

    def open(self):
        """ Energizes valve if Normally Closed. De-energizes if Normally Open."""
        if self._form:
            self.write(0)
        else:
            self.write(1)

    def close(self):
        """ De-energizes valve if Normally Closed. Energizes if Normally Open."""
        if self.form == 'Normally Closed':
            self.write(0)
        else:
            self.write(1)

    @property
    def is_open(self) -> bool:
        """ Implements parent's abstractmethod; returns True if valve is open, False if it is closed"""
        return True if self.read() else False


class PWMControlValve(SolenoidBase, PWMOutput):
    """ An extension of PWMOutput which incorporates linear
    compensation of the valve's response.
    """

    def __init__(self, pin, form='Normally Closed', initial_duty=0, frequency=None, response=None, pig=None):
        """
        Args:
            pin (int): The number of the pin to use
            form (str): The form of the solenoid; can be either `Normally Open` or `Normally Closed`
            initial_duty (float): The initial duty cycle of the pin. Must be between 0 and 1.
            frequency (float): The PWM frequency to use.
            response (str): "/path/to/response/curve/file"
            pig (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        PWMOutput.__init__(self, pin=pin, initial_duty=initial_duty, frequency=frequency, pig=pig)
        SolenoidBase.__init__(self, form=form)
        if response is None:
            raise NotImplementedError('You need to implement a default response behavior')

        self._load_valve_response(response_path=response)

    @property
    def is_open(self) -> bool:
        """ Implements parent's abstractmethod; returns True if valve is open, False if it is closed"""
        if self.setpoint > 0:
            return True
        else:
            return False

    def open(self):
        """ Implements parent's abstractmethod; fully opens the valve"""
        self.setpoint = 1.0

    def close(self):
        """ Implements parent's abstractmethod; fully closes the valve"""
        self.setpoint = 0.0

    @property
    def setpoint(self) -> float:
        """ The linearized setpoint corresponding to the current duty cycle according to the valve's response curve

        Returns:
            float: A number between 0 and 1 representing the current flow as a proportion of maximum
        """
        return self.inverse_response(self.duty)

    @setpoint.setter
    def setpoint(self, setpoint):
        """Overridden to determine & write the duty cycle corresponding
        to the requested linearized setpoint according to the valve's
        response curve

        Args:
            setpoint (float): A number between 0 and 1 representing how much to open the valve
        """
        self.duty = self.response(setpoint)

    def response(self, setpoint, rising=True):
        """Setpoint takes a value in the range (0,1) so as not to confuse with duty cycle, which takes a value in the
        range (0,1). Response curves are specific to individual valves and are to be implemented by subclasses.
        Different curves are calibrated to 'rising = True' (valves opening) or'rising = False' (valves closing), as
        different characteristic flow behavior can be observed.

        Args:
            setpoint (float): A number between 0 and 1 representing how much to open the valve
            rising (bool): Whether or not the requested setpoint is higher than the last (rising = True), or the
                opposite (Rising = False)

        Returns:
            float: The PWM duty cycle corresponding to the requested setpoint
        """

        idx = (np.abs(self._response_array[:, 0] - setpoint)).argmin()
        if rising:
            duty = self._response_array[idx, 1]
        else:
            duty = self._response_array[idx, 2]

        return duty

    def inverse_response(self, duty_cycle, rising=True):

        """Inverse of response. Given a duty cycle in the range (0,1), returns the corresponding linear setpoint in the
        range (0,1).

        Args:
            duty_cycle: The PWM duty cycle
            rising (bool): Whether or not the requested setpoint is higher than the last (rising = True), or the
                opposite (Rising = False)

        Returns:
            float: The setpoint of the valve corresponding to `duty_cycle`
        """
        if rising:
            idx = (np.abs(self._response_array[:, 1] - duty_cycle)).argmin()
        else:
            idx = (np.abs(self._response_array[:, 2] - duty_cycle)).argmin()
        return self._response_array[idx, 0]

    def _load_valve_response(self, response_path):
        """ Loads and applies a response curve of the form `f(setpoint) = duty`. A response curve maps the underlying
        PWM duty cycle `duty` onto the normalized variable `setpoint` representing the flow through the valve as a
        percentage of its maximum.

        Flow through a proportional valve may be nonlinear with respect to [PWM] duty cycle, if the valve itself does
        not include its own electronics to linearize response wrt/ input. Absent on-board compensation of response, a
        proportional solenoid with likely not respond [flow] at all below some minimum threshold duty cycle.
        Above this threshold, the proportional valve begins to open and its response resembles a sigmoid: just past the
        threshold there is a region where flow increases exponentially wrt/ duty cycle, this is followed by a region of
        pseudo-linear response that begins to taper off, eventually approaching the valve's maximum flow asymptotically
        as the duty cycle approaches 100% and the valve opens fully.

        Args:
            response_path: 'path/to/binary/response/file' - if response_path is None, defaults to `setpoint = duty`
        """
        if response_path is not None:
            response_array = np.load(response_path)
        else:
            response_array = np.linspace([0, 0, 0], [1, 1, 1], num=101)
        self._response_array = response_array


class SimOnOffValve(SolenoidBase):
    """ stub: a simulated on/off valve"""

    def __init__(self, form='Normally Closed', pig=None):
        super().__init__(form=form)
        self.state = 0

    def open(self):
        self.state = 1

    def close(self):
        self.state = 0

    @property
    def is_open(self):
        return True if self.state else False


class SimControlValve(SolenoidBase):
    """stub: a simulated linear control valve"""

    @property
    def is_open(self) -> bool:
        """ Implements parent's abstractmethod; returns True if valve is open, False if it is closed"""
        if self.setpoint > 0:
            return True
        else:
            return False

    def open(self):
        """ Implements parent's abstractmethod; fully opens the valve
        FIXME: Needs refactor; duplicate method to PWMControlValve.open()"""
        self.setpoint = 1.0

    def close(self):
        """ Implements parent's abstractmethod; fully closes the valve
        FIXME: Needs refactor; duplicate method to PWMControlValve.close()"""
        self.setpoint = 0.0

    def __init__(self, form='Normally Closed', pig=None):
        """
        Args:
            form (str): The form of the solenoid; can be either `Normally Open` or `Normally Closed`
            pig (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        super().__init__(form=form)
        self._setpoint = 0

    @property
    def setpoint(self):
        """ The requested linearized set-point of the valve.

        Returns:
            float: A number between 0 and 1 representing the current flow as a proportion of maximum

        """
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value):
        """
        Args:
            value: A float between 0 and 1; the requested set-point of the valve as a proportion of maximum

        """
        if not 0 <= value <= 1:
            raise ValueError('Setpoint must be in [0, 1]')
        else:
            self._setpoint = value
