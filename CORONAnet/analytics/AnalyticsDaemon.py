"""
Defines a daemon for periodically checking run directory
for updated summaries files. If the summaries files have been
updated, generate new plots

Author: Peter Thomas
Date: 01 May 2022
"""
import os
import sys
import time
import atexit
import signal
import hashlib
import argparse
from datetime import datetime
from .read_summaries import run_analysis


def sha1(filename):
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


class AnalyticsDaemon:
    """
    Instantiates daemon to periodically look through analysis directory and
    update plots when changes to summary files have been detected
    """
    def __init__(self,
                 pidfile, 
                 analytics_root_dir,
                 stdin='/dev/null', 
                 stdout='/dev/null', 
                 stderr='/dev/null',
                 delay_time=60):
        self.stdin = stdin
        self.stdout = stdout 
        self.stderr = stderr
        self.pidfile = pidfile
        self.delay_time = delay_time
        self.analytics_root_dir = analytics_root_dir

        # Keep track of log file hashes
        self.latest_hash_list = None

        # flag to set if we are running analysis loop
        self.is_running = False

    def daemonize(self):
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent 
                sys.exit(0)
        except OSError as e:
            sys.stderr.write("fork #1 failed: {0}\n".format(e))
            sys.exit(1)

        # decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                sys.exit(0)

        except OSError as e:
            sys.stderr.write("fork #2 failed: {0}\n".format(e))
            sys.exit(1)

        # redirect standard file descriptors 
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile
        atexit.register(self.delpid)

        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as f:
            f.write(pid + '\n')

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):
        """
        Start the daemon.
        """
        # check for a pidfile to see if the daemon already runs 
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None 

        if pid:
            message = "pidfile {0} already exists. " + \
                    "Daemon already running?\n"
            sys.stderr.write(message.format(self.pidfile))
            sys.exit(1)

        # start the daemon 
        self.daemonize()

        # set run flag 
        self.is_running = True

        # run analysis loop
        self.run()

    def stop(self):
        """
        Stop the daemon 
        """
        # Get the pid from the pidfile 
        try:
            with open(self.pidfile, 'r') as pf:
                pid = int(pf.read().strip())
        except IOError:
            pid = None

        if not pid: 
            message = "pidfile {0} does not exist. " + \
                    "Daemon not running?\n"
            sys.stderr.write(message.format(self.pidfile))
            return # not an error in a restart

        # set run loop flag to 'False'
        self.is_running = False

        # Try killing the daemon process 
        try:
            while True:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
        except OSError as err:
            e = str(err.args)
            if e.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print(str(err.args))
                sys.exit(1)

    def restart(self):
        """
        Restart the daemon
        """
        self.stop()
        self.start()

    def run(self):
        """
        Perform analytics functions
        """
        while self.is_running:
            # search the analytics root directory for log files                
            serialized_log_files = list()

            for root, dirs, files in os.walk(self.analytics_root_dir):
                for file in files:
                    if 'events.out.tfevents' in file:
                        serialized_log_files.append(os.path.join(root, file))

            # Get hashes for files
            latest_hashes = list()
            hashes_have_changed = False
            for log_file in serialized_log_files:
                hash = sha1(log_file)

                # if the hash is not in the currently stored list of hashes, 
                # the file has changed
                if self.latest_hash_list is None or hash not in self.latest_hash_list:
                    hashes_have_changed = True

                latest_hashes.append(hash)

            # update the hash list
            self.latest_hash_list = latest_hashes

            # If we detected a change, run analytics
            if hashes_have_changed:
                run_analysis(self.analytics_root_dir)

            # sleep until we want to check analysis directory again
            time.sleep(self.delay_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--analytics_root_dir', 
                        type=str,
                        required=True, 
                        help="Root directory to perform run analysis")

    parser.add_argument('--delay_time',
                        type=int,
                        default=600,
                        help="Time to wait until we check root directory for changes again")

    # add mutually exclusive arguments to start, stop, and restart daemon
    daemon_functions_group = parser.add_mutually_exclusive_group(required=True)

    daemon_functions_group.add_argument('--start', 
                                        action='store_true',
                                        help="Start analysis daemon")

    daemon_functions_group.add_argument('--stop',
                                        action='store_true',
                                        help="Stop analysis daemon")

    daemon_functions_group.add_argument('--restart',
                                        action='store_true',
                                        help="Restart analysis daemon")

    flags = parser.parse_args()

    # instantiate analysis daemon
    daemon = AnalyticsDaemon('/tmp/analysis-daemon.pid', 
                             flags.analytics_root_dir, 
                             delay_time=flags.delay_time)

    if flags.start:
        daemon.start()

    elif flags.stop:
        daemon.stop()

    elif flags.restart:
        daemon.restart()
