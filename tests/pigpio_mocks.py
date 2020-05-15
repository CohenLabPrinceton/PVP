import random
import pytest
import functools
from collections import deque
import pigpio
from vent.io.devices.pins import Pin

swap_dict = {
    "NORMAL": "SWAPPED",
    "SWAPPED": "NORMAL"
}
reg_nums = {'CONVERSION': 0, 'CONFIG': 1, 'SFM': 2}
reg_names = ('CONVERSION', 'CONFIG', 'SFM')
soft_frequencies = (8000, 4000, 2000, 1600, 1000, 800, 500, 400, 320, 250, 200, 160, 100, 80, 50, 40, 20, 10)


@pytest.fixture()
def mock_pigpio_base(monkeypatch):
    monkeypatch.setattr("socket.create_connection", mock_create_connection)
    monkeypatch.setattr("pigpio._socklock", MockSockLock)
    monkeypatch.setattr("pigpio._callback_thread", MockThread)
    monkeypatch.setattr("pigpio._pigpio_command", mock_pigpio_command)
    monkeypatch.delattr("pigpio._pigpio_command_nolock")
    monkeypatch.delattr("pigpio._pigpio_command_ext")
    monkeypatch.delattr("pigpio._pigpio_command_ext_nolock")
    monkeypatch.setattr("pigpio.pi.connected", 1, raising=False)


@pytest.fixture()
def mock_pigpio_i2c(mock_pigpio_base, monkeypatch):
    monkeypatch.setattr("pigpio.pi.mock_i2c", {0: dict(), 1: dict()}, raising=False)
    monkeypatch.setattr("pigpio.pi.mock_handles", dict(), raising=False)
    monkeypatch.setattr("pigpio.pi.add_mock_i2c_hardware", add_mock_i2c_hardware, raising=False)
    monkeypatch.setattr("pigpio.pi.i2c_open", mock_i2c_open)
    monkeypatch.setattr("pigpio.pi.i2c_close", mock_i2c_close)
    monkeypatch.setattr("pigpio.pi.i2c_read_device", mock_i2c_read_device)
    monkeypatch.setattr("pigpio.pi.i2c_read_i2c_block_data", mock_i2c_read_i2c_block_data)
    monkeypatch.setattr("pigpio.pi.i2c_write_device", mock_i2c_write_device)
    monkeypatch.setattr("pigpio.pi.i2c_write_i2c_block_data", mock_i2c_write_i2c_block_data)


@pytest.fixture()
def mock_pigpio_gpio(mock_pigpio_base, monkeypatch):
    """ Monkeypatches pigpio.pi with a mock GPIO interface.
    - Creates a new list "pigpio.pi.mock_pins" of MockPigpioPin objects, one for each RPi GPIO pin.
    - monkeypatches pigpio.pi.get_mode() to look at pigpio.pi.mock_pins for the mode of each pin instead of
        communicating with the pigpiod daemon.
    - monkeypatches pigpio.pi.set_mode() in a similar manner

    Args:
        mock_pigpio_base: monkeypatches necessary for initializing pigpio
    """
    monkeypatch.setattr("pigpio.pi.mock_pins", [MockPigpioPin(num) for num in range(53)], raising=False)
    monkeypatch.setattr("pigpio.pi.get_mode", mock_pigpio_get_mode)
    monkeypatch.setattr("pigpio.pi.set_mode", mock_pigpio_set_mode)
    monkeypatch.setattr("pigpio.pi.read", mock_pigpio_read)
    monkeypatch.setattr("pigpio.pi.write", mock_pigpio_write)
    monkeypatch.setattr("pigpio.pi.get_PWM_frequency", mock_get_PWM_frequency)
    monkeypatch.setattr("pigpio.pi.set_PWM_frequency", mock_set_PWM_frequency)
    monkeypatch.setattr("pigpio.pi.get_PWM_dutycycle", mock_get_PWM_dutycycle)
    monkeypatch.setattr("pigpio.pi.set_PWM_dutycycle", mock_set_PWM_dutycycle)
    monkeypatch.setattr("pigpio.pi.get_PWM_range", mock_get_PWM_range)
    monkeypatch.setattr("pigpio.pi.hardware_PWM", mock_hardware_PWM)


def mock_pigpio_errors(func):
    @functools.wraps(func)
    def mock_pigpio__u2i_exception(self, *args, **kwargs):
        value = func(self, *args, **kwargs)
        if value < 0:
            raise pigpio.error(pigpio.error_text(value))
        return value
    return mock_pigpio__u2i_exception


def mock_pigpio_bad_gpio_arg(func):
    @functools.wraps(func)
    def check_args(self, gpio, *args, **kwargs):
        if gpio in range(53):
            result = func(self, gpio, *args, **kwargs)
        else:
            result = pigpio._errors[pigpio.PI_BAD_GPIO]
        return result
    return check_args


def mock_pigpio_bad_user_gpio_arg(func):
    @functools.wraps(func)
    def check_args(self, gpio, *args, **kwargs):
        if gpio in range(31):
            result = func(self, gpio, *args, **kwargs)
        else:
            result = pigpio._errors[-2]
        return result
    return check_args


def mock_register(reg, swp, idx = None):
    """ A collection of fake register values. Values returned are values that might actually be found in the registers
    they claim to be from.

    Args:
        reg: Which fake register the values might have come from. (From the ADS1x15: 'CONFIG', 'CONVERSION'. From the
            SFM3200: 'SFM')
        swp: 'NORMAL' or byteswapped: 'SWAPPED'
        idx: index of value. 0 & 1 are normal values for testing purposes, reg, swp = 'CONFIG', 2 contains a value that
            might be found on the ADS1x15 when a conversion has begun but is not ready yet. (OS bit = 0)

    Returns:
        int: a register value
    """
    registers = {
        'CONVERSION': {
            'NORMAL': (0x682A, 0x223F),
            'SWAPPED': (0x2A68, 0x3F22)
        },
        'CONFIG': {
            'NORMAL': (0xC3E3, 0x8583, 0x43E3),
            'SWAPPED': (0xE3C3, 0x8385, 0xE343)
        },
        'SFM': {
            'NORMAL': (0x8E10, 0x71F0),
            'SWAPPED': (0x108E, 0xF071)
        }
    }
    if idx is None:
        return registers[reg][swp]
    else:
        return registers[reg][swp][idx]


class MockThread:
    def __init__(self, *args, **kwargs):
        pass

    def stop(self):
        pass


class MockSocket:
    """ Bare-bones mock of socket.Socket with additional mocks of functions called by pigpio"""
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def close():
        pass

    def setsockopt(self, *args, **kwargs):
        pass


class MockSockLock:
    """ Bare-bones mock of pigpio._socklock"""
    def __init__(self, *args, **kwargs):
        self.s = None
        self.l = None


def mock_pigpio_command(*args, **kwargs):
    """ Make sure no commands get sent to the pigpio daemon"""
    pass


def mock_create_connection(host, timeout):
    """ mock of socket.create_connection(). Returns a bare-bones mock socket"""
    return MockSocket


class MockI2CHardwareDevice:
    def __init__(self, *args):
        """ A simple mock device with registers defined by kwargs. If only one register value is passed, device emulates
        a single register device that responds to read_device commands (such as the SFM3200). Retains past register
        values upon writing to a register for logging/assertion convenience.

        Args:
            *args: register_values; one, per, register
        """
        self.last_register = None
        self.registers = []
        if args:
            for arg in args:
                if isinstance(arg, int):
                    arg = arg.to_bytes(2, 'little')
                elif not isinstance(arg, bytes):
                    raise TypeError("Unknown register value of type {}".format(type(arg)))
                self.registers.append(deque())
                self.registers[-1].append(arg)
            if len(self.registers) == 1:
                self.last_register = 0
        else:
            TypeError("MockI2CHardwareDevice needs at least one register to initialize")

    def read_device(self, count=None):
        """Alias for read_register(reg_num=None, count)"""
        return self.read_register(reg_num=None, count=count)

    def write_device(self, value):
        """ Alias for write_register(reg_num=None, value)"""
        self.write_register(reg_num=None, value=value)

    def read_register(self, reg_num=None, count=None):
        """ Reads count bytes from a specific register

        Args:
            reg_num: The index of the register to read
            count (int): The number of bytes to read

        Returns:
            bytes: the register contents
        """
        if reg_num is None:
            if self.last_register is not None:
                reg_num = self.last_register
            else:
                raise RuntimeError("mock_i2c_device tried to access last register but none have been accessed yet")
        else:
            self.last_register = reg_num
        result = self.registers[reg_num][-1]
        return result if count is None else result[:count]

    def write_register(self, reg_num, value):
        """ Writes value to register specified by reg_num

        Args:
            reg_num: The index of the register to write
            value (bytes): The stuff to write to the register
        """
        if type(value) is not bytes:
            raise TypeError("arg 'value' must be of type bytes")
        if reg_num is None:
            if self.last_register is not None:
                reg_num = self.last_register
            else:
                raise RuntimeError("mock_i2c_device tried to access last register but none have been accessed yet")
        else:
            self.last_register = reg_num
        self.last_register = reg_num
        self.registers[reg_num].append(value)


def add_mock_i2c_hardware(self, device: MockI2CHardwareDevice, i2c_address, i2c_bus=1):
    self.mock_i2c[i2c_bus][i2c_address] = device


@mock_pigpio_errors
def mock_i2c_open(self, i2c_bus, i2c_address):
    if i2c_bus in self.mock_i2c:
        handle = len(self.mock_i2c[i2c_bus])
        self.mock_handles = {handle: (i2c_bus, i2c_address)}
        return handle
    else:
        return pigpio.PI_BAD_I2C_BUS


@mock_pigpio_errors
def mock_i2c_close(self, handle):
    if handle in self.mock_handles:
        (i2c_bus, i2c_address) = self.mock_handles[handle]
        del self.mock_i2c[i2c_bus][i2c_address]
        del self.mock_handles[handle]
        return 0
    else:
        return pigpio.PI_BAD_HANDLE


@mock_pigpio_errors
def mock_i2c_read_device(self, handle, count):
    if handle in self.mock_handles:
        (bus, addr) = self.mock_handles[handle]
        if count != 2:
            raise NotImplementedError
        return count, self.mock_i2c[bus][addr].read_device(count)
    else:
        return (pigpio.PI_BAD_HANDLE, )


@mock_pigpio_errors
def mock_i2c_read_i2c_block_data(self, handle, reg, count):
    if handle in self.mock_handles:
        (bus, addr) = self.mock_handles[handle]
        if count != 2:
            raise NotImplementedError
        return count, self.mock_i2c[bus][addr].read_register(reg, count)
    else:
        return (pigpio.PI_BAD_HANDLE, )


@mock_pigpio_errors
def mock_i2c_write_device(self, handle, data):
    if handle in self.mock_handles:
        (bus, addr) = self.mock_handles[handle]
        if type(data) is int:
            data = data.to_bytes(2, 'little')
        elif type(data) is bytes:
            data = bytes(reversed(data))
        else:
            raise NotImplementedError
        self.mock_i2c[bus][addr].write_device(data)
        return 0
    else:
        return pigpio.PI_BAD_HANDLE


@mock_pigpio_errors
def mock_i2c_write_i2c_block_data(self, handle, reg, data):
    if handle in self.mock_handles:
        (bus, addr) = self.mock_handles[handle]
        if type(data) is int:
            data = data.to_bytes(2, 'little')
        elif type(data) is bytes:
            data = bytes(reversed(data))
        else:
            raise NotImplementedError
        self.mock_i2c[bus][addr].write_register(reg, data)
        return 0
    else:
        return pigpio.PI_BAD_HANDLE


class MockPigpioPin:
    def __init__(self, gpio: int):
        """ A simple object to mock a pigpio GPIO pin interface without communicating with the pigpiod daemon. It can
        be determined whether PWM is in use on the pin by checking if self.pwm_duty_cycle is None.

        Note: PWM frequency is weird. Basically, starting a hardware PWM with some frequency does not change the
            underlying (soft) PWM frequency. If the mode on the pin changes, the PWM frequency will revert to that
            soft frequency. However, while the hardware PWM is active, the hardware PWM frequency can be modified by
            calling set_PWM_frequency without triggering a change in mode - this DOES change the underlying soft PWM
            frequency, and will behave as though only soft frequencies are allowed. You can only set a PWM frequency
            outside of those allowed by soft_frequencies via starting a hardware PWM.

        Args:
            gpio (int): A number between 0 and 53
        """
        self.errors = dict(pigpio._errors)
        assert gpio in range(53)
        self._mode = random.choice([*Pin._PIGPIO_MODES.values()])
        self.gpio = gpio
        self.level = 0
        self.soft_pwm_frequency = 800
        self.hard_pwm_frequency = None
        self.pwm_duty_cycle = None

    @property
    def pwm_range(self):
        return 1000000 if self.mode == 4 else 255

    @property
    def pwm_frequency(self):
        return self.hard_pwm_frequency if self.mode == 4 else self.soft_pwm_frequency

    @pwm_frequency.setter
    def pwm_frequency(self, pwm_frequency):
        if self.mode == 4:
            self.hard_pwm_frequency = pwm_frequency
        self.soft_pwm_frequency = pwm_frequency

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        """ Seems redundant with mock_pigpio_set_mode, but it isn't because mode can be set as a side effect of calling
        other methods, such as starting a PWM. Also, mode MUST be changed prior to setting a pwm_duty_cycle otherwise
        it will be unintentionally wiped
        """
        if new_mode != self.mode and self.pwm_duty_cycle is not None:
            self.pwm_duty_cycle = None
        self._mode = new_mode


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_pigpio_get_mode(self, gpio):
    """ Returns the mode if arg gpio is valid, otherwise mocks the pigpio errnum for PI_BAD_GPIO
    Args:
        self (pigpio.pi): A pigpio.pi or vent.io.devices.PigpioConnection instance
        gpio (int): A number between 0 and 53

    Returns:
        int: A number between 0 and 7
    """
    return self.mock_pins[gpio].mode


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_pigpio_set_mode(self, gpio, mode):
    """ Will stop a PWM if new mode != mode (via property setter), but not if new mode is the same as set mode
    Args:
        mode (int): A number between 0 and 7 (See: vent.io.devices.pins.Pin._PIGPIO_MODES)

    Returns:
        int: 0 if successful
    """
    if mode not in range(8):
        return self.mock_pins[gpio].errors[-4]
    else:
        self.mock_pins[gpio].mode = mode
        return 0


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_pigpio_read(self, gpio):
    """ Side effect: Will always set mode to 0 (which will always wipe a PWM)

    Returns:
        int: 0 if pin is pulled low, 1 if pin is pulled high
    """
    self.mock_pins[gpio].mode = 0
    return self.mock_pins[gpio].level


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_pigpio_write(self, gpio, level):
    """ Side effect: Will always wipe a PWM, and will set self.mode
    Args:
        level (int): 0 if pin is to be pulled low, 1 if pin is to be pulled high

    Returns:
        int: 0 if successful
    """
    if level not in range(2):
        return self.mock_pins[gpio].errors[-5]
    else:
        self.mock_pins[gpio].pwm_dutycycle = None
        self.mock_pins[gpio].mode = level
        self.mock_pins[gpio].level = level
        return 0


@mock_pigpio_bad_user_gpio_arg
@mock_pigpio_errors
def mock_get_PWM_frequency(self, gpio):
    """ Note: Will return soft frequency if mode is not 4, regardless of whether a PWM is in use. If mode == 4, returns the
    hardware PWM frequency. This is handled via property getter for self.pwm_frequency

    Returns
        int: The frequency used for PWM on the GPIO, in Hz
    """
    result = self.mock_pins[gpio].pwm_frequency
    if result is not None:
        return result
    else:
        return self.mock_pins[gpio].errors[-92]


@mock_pigpio_bad_user_gpio_arg
@mock_pigpio_errors
def mock_set_PWM_frequency(self, gpio, PWMfreq):
    """
    Args:
        PWMfreq (int): The frequency to be set on the pin.

    Returns:
        int: The frequency set on the GPIO. If PWMfreq not in soft_frequencies, it is the closest allowed frequency
    """
    if PWMfreq not in soft_frequencies:
        PWMfreq = min(soft_frequencies, key=lambda x: abs(x-PWMfreq))
    self.mock_pins[gpio].pwm_frequency = PWMfreq
    return PWMfreq


@mock_pigpio_bad_user_gpio_arg
@mock_pigpio_errors
def mock_get_PWM_dutycycle(self, gpio):
    """
    Returns:
        int: The duty cycle used for the GPIO, out of the PWM range (default 255)
    """
    result = self.mock_pins[gpio].pwm_duty_cycle
    return result if result is not None else self.errors[-92]


@mock_pigpio_bad_user_gpio_arg
@mock_pigpio_errors
def mock_set_PWM_dutycycle(self, gpio, PWMduty):
    """ Note: This will start a soft PWM and change pin mode if mode is not already 1. Must set self.mode prior
    to setting self.duty_cycle

    Args:
        PWMduty (int): the duty cycle to be use for the GPIO, must be in range(pwm_range+1)

    Returns:
        int: 0 if successful
    """

    if PWMduty in range(self.mock_pins[gpio].pwm_range+1):
        self.mode = 1
        self.mock_pins[gpio].pwm_duty_cycle = PWMduty
        return 0
    else:
        return self.mock_pins[gpio].errors[-8]


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_get_PWM_range(self, gpio):
    """
    Returns:
        int: 255 unless hardware PWM then 1e6
    """
    return self.mock_pins[gpio].pwm_range


@mock_pigpio_bad_gpio_arg
@mock_pigpio_errors
def mock_hardware_PWM(self, gpio, PWMfreq, PWMduty):
    """ Returns appropriate pigpio error if GPIO, PWMfreq, or PWMduty are not in their respective allowable ranges for
    pigpio hardware PWM.
    Note: must set self.pwm_duty_cycle prior to setting self.mode!

    Returns:
        int: 0 if successful
    """
    self.mock_pins[gpio].mode = 4
    if gpio not in (12, 13, 18, 19):
        return self.mock_pins[gpio].errors[-95]
    elif PWMfreq > 187500000 or PWMfreq < 0:
        return self.mock_pins[gpio].errors[-96]
    elif PWMduty not in range(self.mock_pins[gpio].pwm_range + 1):
        return self.mock_pins[gpio].errors[-97]
    else:
        self.mock_pins[gpio].hard_pwm_frequency = PWMfreq
        self.mock_pins[gpio].pwm_duty_cycle = PWMduty
        return 0
