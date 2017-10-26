import sys
import time
import datetime

class ProgressBar(object):
    def __init__(self, num_iterations, length=40, absolute_numbers=True, unit=""):
        self.length = length
        self.absolute_numbers = absolute_numbers
        self.unit = unit
        self.start_time = time.time()
        self.last_time = time.time()
        self.num_iterations = num_iterations   
        self.time_per_iteration = 0.0
        self.a = 0.5


    def percent(self, i):
        return float(i+1) / self.num_iterations


    def __call__(self, i, message=None):
        percent = self.percent(i)
        dots = int(percent * self.length)
        iterations_left = self.num_iterations - i - 1
        time_for_last_iteration = time.time() - self.last_time
        self.last_time = time.time()
        self.time_per_iteration = self.a * time_for_last_iteration + (1.0 - self.a) * self.time_per_iteration
        time_remaining = self.time_per_iteration * iterations_left
        minutes_remaining = int(time_remaining) // 60
        seconds_remaining = int(time_remaining) % 60
        eta_str = "%02d:%02d" % (minutes_remaining, seconds_remaining)


        if percent < 1.0:
            bar_length = max(dots - 1,0)
            bar = "[" + '='*(bar_length) + '>' + '.'*(self.length - bar_length - 1) + ']'
        else:
            bar = '[' + '='*self.length + ']'

        bar += "  %d/%d %s" % (i+1,self.num_iterations,self.unit)
        
        if percent < 1.0:
            bar += " ETA: %s" % eta_str
        else:
            total_time = int(time.time() - self.start_time) 
            bar += " TIME: %02d:%02d" % (total_time//60, total_time % 60)

        sys.stdout.write('\r' + bar + " " + message)
        sys.stdout.flush()
        if i == self.num_iterations - 1:
            print("")
