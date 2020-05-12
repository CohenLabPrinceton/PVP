from itertools import count


class Alarm:
    """
    Class used by the program to control and coordinate alarms.

    Parameterized by a :class:`Alarm_Rule` and managed by :class:`Alarm_Manager`
    """


    id_counter = count()
    """
    :class:`itertools.count`: used to generate unique IDs for each alarm
    """

    def __init__(self, alarm_name, is_active, severity, alarm_start_time, alarm_end_time,value=None, message=None):
        """

        Args:
            alarm_name :
            is_active:
            severity:
            alarm_start_time:
            alarm_end_time:
            value (int, float): optional - numerical value that generated the alarm
            message (str): optional - override default text generated by :class:`~vent.gui.alarm_manager.AlarmManager`
        """
        self.id = next(Alarm.id_counter)
        self.alarm_name = alarm_name
        self.is_active = is_active
        self.severity = severity
        self.alarm_start_time = alarm_start_time
        self.alarm_end_time = alarm_end_time
        self.value = value
        self.message = message