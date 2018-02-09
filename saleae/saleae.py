#!/usr/bin/env python3
# vim: tw=80 ts=4 sts=4 sw=4 smarttab noet


# Coerce Py2k to act more like Py3k
# https://pypi.python.org/pypi/future

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import (bytes, dict, int, list, object, range, str, ascii, chr,
		hex, input, next, oct, open, pow, round, super, filter, map, zip)


import logging
log = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

import bisect
import contextlib
import enum
import inspect
import os
import platform
import psutil
import shutil
import socket
import sys
import time
import warnings

# Py2k compat. This isn't actually 1:1, but is sufficient for our purposes
try:
	ConnectionRefusedError
except NameError:
	ConnectionRefusedError = socket.error

PY2K = sys.version_info[0] == 2
PY3K = sys.version_info[0] == 3

@enum.unique
class Trigger(enum.IntEnum):
	# Python convention is to start enums at 1 for truth checks, but it
	# seems reasonable that no trigger should compare as false
	NoTrigger = 0
	High = 1
	Low = 2
	Posedge = 3
	Negedge = 4

@enum.unique
class PerformanceOption(enum.IntEnum):
	Full = 100
	Half = 50
	Third = 33
	Quarter = 25
	Low = 20

class ConnectedDevice():
	def __init__(self, type, name, id, index, active):
		self.type = type
		self.name = name
		self.id = int(id, 16)
		self.index = int(index)
		self.active = bool(active)

	def __str__(self):
		if self.active:
			return "<saleae.ConnectedDevice #{self.index} {self.type} {self.name} ({self.id:x}) **ACTIVE**>".format(self=self)
		else:
			return "<saleae.ConnectedDevice #{self.index} {self.type} {self.name} ({self.id:x})>".format(self=self)

	def __repr__(self):
		return str(self)


class Saleae():
	class SaleaeError(Exception):
		pass

	class CommandNAKedError(SaleaeError):
		pass

	class ImpossibleSettings(SaleaeError):
		pass

	@staticmethod
	def launch_logic(timeout=5):
		'''Attempts to open Saleae Logic software'''
		if platform.system() == 'Darwin':
			ret = os.system('open /Applications/Logic.app')
			if ret != 0:
				raise OSError("Failed to open Logic software")
		elif platform.system() == 'Linux':
			if PY2K:
				log.warn("PY2K support limited. If `Logic` is not on your PATH it will not open.")
				os.system("Logic &")
			else:
				path = shutil.which('Logic')
				if path is None:
					raise OSError("Cannot find Logic software. Is 'Logic' in your PATH?")
				os.system(path + '&')
		elif platform.system() == 'Windows':
			p = os.path.join("C:", "Program Files", "Saleae Inc", "Logic.exe")
			if not os.path.exists(p):
				p = os.path.join("C:", "Program Files", "Saleae LLC", "Logic.exe")
			os.startfile(p)
		else:
			raise NotImplementedError("Unknown platform " + platform.system())

		# Try to intelligently wait for Logic to be ready, but can't wait
		# forever because user may not have enabled the scripting server
		while timeout > 0:
			with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
				if sock.connect_ex(('localhost', 10429)) == 0:
					break
			log.debug('launch_logic: port not yet open, sleeping 1s')
			time.sleep(1)
			timeout -= 1

	@staticmethod
	def kill_logic():
		'''Attempts to find and kill running Saleae Logic software'''
		# This is a bit experimental as I'm not sure what the process name will
		# be on every platform. For now, I'm making the hopefully reasonable and
		# conservative assumption that if there's only one process running with
		# 'logic' in the name, that it's Saleae Logic
		candidates = []
		for proc in psutil.process_iter():
			try:
				if 'logic' in proc.name().lower():
					candidates.append(proc)
			except psutil.NoSuchProcess:
				pass
		if len(candidates) == 0:
			raise OSError("No logic process found")
		if len(candidates) > 1:
			raise NotImplementedError("Multiple candidates for logic software."
					" Not sure which to kill: " + str(candidates))
		candidates[0].terminate()

	def __init__(self, host='localhost', port=10429):
		self._to_send = []
		self.sample_rates = None
		self.connected_devices = None

		try:
			self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._s.connect((host, port))
		except ConnectionRefusedError:
			log.info("Could not connect to Logic software, attempting to launch it now")
			Saleae.launch_logic()

		try:
			self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._s.connect((host, port))
		except ConnectionRefusedError:
			print("Failed to connect to saleae at {}:{}".format(host, port))
			print("")
			print("Did you remember to 'Enable scripting socket server' (see README)?")
			print("")
			raise
		log.info("Connected.")
		self._rxbuf = ''

	def _build(self, s):
		'''Convenience method for building up a command to send'''
		if isinstance(s, list):
			self._to_send.extend(s)
		else:
			self._to_send.append(s)

	def _abort(self):
		self._to_send = []

	def _finish(self, s=None):
		if s:
			self._build(s)
		try:
			ret = self._cmd(', '.join(self._to_send))
		finally:
			self._to_send = []
		return ret

	def _round_up_or_max(self, value, candidates):
		i = bisect.bisect_left(candidates, value)
		if i == len(candidates):
			i -= 1
		return candidates[i]

	def _send(self, s):
		log.debug("Send >{}<".format(s))
		self._s.send(bytes(s + '\0', 'UTF-8'))

	def _recv(self, expect_nak=False):
		while 'ACK' not in self._rxbuf:
			self._rxbuf += self._s.recv(1024).decode('UTF-8')
			log.debug("Recv >{}<".format(self._rxbuf))
			if 'NAK' == self._rxbuf[0:3]:
				self._rxbuf = self._rxbuf[3:]
				if expect_nak:
					return None
				else:
					raise self.CommandNAKedError
		ret, self._rxbuf = self._rxbuf.split('ACK', 1)
		return ret

	def _cmd(self, s, wait_for_ack=True, expect_nak=False):
		self._send(s)

		ret = None
		if wait_for_ack:
			ret = self._recv(expect_nak=expect_nak)
		return ret

	def set_trigger_one_channel(self, digital_channel, trigger):
		'''Convenience method to set one trigger.

		:param channel: Integer specifying channel
		:param trigger: saleae.Trigger indicating trigger type
		:raises ImpossibleSettings: rasied if channel is not active
		'''
		digital, analog = self.get_active_channels()

		to_set = [Trigger.NoTrigger for x in range(len(digital))]
		trigger = Trigger(trigger)
		try:
			to_set[digital.index(digital_channel)] = trigger
		except ValueError:
			raise self.ImpossibleSettings("Cannot set trigger on inactive channel")
		self._set_triggers_for_all_channels(to_set)

	def _set_triggers_for_all_channels(self, channels):
		self._build('SET_TRIGGER')
		for c in channels:
			# Try coercing b/c it will throw a nice exception if it fails
			c = Trigger(c)
			if c == Trigger.NoTrigger:
				self._build('')
			elif c == Trigger.High:
				self._build('high')
			elif c == Trigger.Low:
				self._build('low')
			elif c == Trigger.Posedge:
				self._build('posedge')
			elif c == Trigger.Negedge:
				self._build('negedge')
			else:
				raise NotImplementedError("Must pass trigger type")
		self._finish()

	def set_triggers_for_all_channels(self, channels):
		'''Set the trigger conditions for all active digital channels.

		:param channels: An array of saleae.Trigger for each channel
		:raises ImpossibleSettings: rasied if configuration is not provided for all channels

		*Note: Calls to this function must always set all active digital
		channels. The Saleae protocol does not currently expose a method to read
		current triggers.*'''

		digital, analog = self.get_active_channels()
		if len(channels) != len(digital):
			raise self.ImpossibleSettings("Trigger settings must set all active digital channels")

		self._set_triggers_for_all_channels(channels)

	def set_num_samples(self, samples):
		'''Set the capture duration to a specific number of samples.

		:param samples: Number of samples to capture, will be coerced to ``int``

		*From Saleae documentation*
		  Note: USB transfer chunks are about 33ms of data so the number of
		  samples you actually get are in steps of 33ms.

		>>> s.set_num_samples(1e6)
		'''
		self._cmd('SET_NUM_SAMPLES, {:d}'.format(int(samples)))

	def set_capture_seconds(self, seconds):
		'''Set the capture duration to a length of time.

		:param seconds: Capture length. Partial seconds (floats) are fine.

		>>> s.set_capture_seconds(1)
		'''
		self._cmd('SET_CAPTURE_SECONDS, {}'.format(float(seconds)))

	def set_sample_rate(self, sample_rate_tuple):
		'''Set the sample rate. Note the caveats. Consider ``set_sample_rate_by_minimum``.

		Due to saleae software limitations, only sample rates exposed in the
		Logic software can be used. Use the ``get_all_sample_rates`` method to
		get all of the valid sample rates. The list of valid sample rates
		changes based on the number and type of active channels, so set up all
		channel configuration before attempting to set the sample rate.

		:param sample_rate_tuple: A sample rate as returned from ``get_all_sample_rates``

		>>> s.set_sample_rate(s.get_all_sample_rates()[0])
		'''

		self.get_all_sample_rates()
		if sample_rate_tuple not in self.sample_rates:
			raise NotImplementedError("Unsupported sample rate")

		self._cmd('SET_SAMPLE_RATE, {}, {}'.format(*sample_rate_tuple))

	def set_sample_rate_by_minimum(self, digital_minimum=0, analog_minimum=0):
		'''Set to a valid sample rate given current configuration and a target.

		Because the avaiable sample rates are not known until runtime after all
		other configuration, this helper method takes a minimum digital and/or
		analog sampling rate and will choose the minimum sampling rate available
		at runtime. Setting digital or analog to 0 will disable the respective
		sampling method.

		:param digital_minimum: Minimum digital sampling rate in samples/sec or 0 for don't care
		:param analog_minimum: Minimum analog sampling rate in samples/sec or 0 for don't care
		:returns (digital_rate, analog_rate): the sample rate that was set
		:raises ImpossibleSettings: rasied if sample rate cannot be met

		>>> s.set_sample_rate_by_minimum(1e6, 1)
		(12000000, 10)
		'''

		if digital_minimum == analog_minimum == 0:
			raise self.ImpossibleSettings("One of digital or analog minimum must be nonzero")

		self.get_all_sample_rates()

		# Sample rates may be unordered, iterate all tracking the best
		best_rate = None
		best_bandwidth = None
		for rate in self.sample_rates:
			if digital_minimum != 0 and digital_minimum <= rate[0]:
				if (analog_minimum == 0) or (analog_minimum != 0 and analog_minimum <= rate[1]):
					if best_rate is None:
						best_rate = rate
						best_bandwidth = self.get_bandwidth(sample_rate=rate)
					else:
						new_bandwidth = self.get_bandwidth(sample_rate=rate)
						if new_bandwidth < best_bandwidth:
							best_rate = rate
							best_bandwidth = new_bandwidth
			elif analog_minimum != 0 and analog_minimum <= rate[1]:
				if best_rate is None:
					best_rate = rate
					best_bandwidth = self.get_bandwidth(sample_rate=rate)
				else:
					new_bandwidth = self.get_bandwidth(sample_rate=rate)
					if new_bandwidth < best_bandwidth:
						best_rate = rate
						best_bandwidth = new_bandwidth

		if best_rate is None:
			raise self.ImpossibleSettings("No sample rate for configuration. Try lowering rate or disabling channels (especially analog channels)")

		self.set_sample_rate(best_rate)
		return best_rate

	def get_all_sample_rates(self):
		'''Get available sample rate combinations for the current performance level and channel combination.

		>>> s.get_all_sample_rates()
		[(12000000, 6000000), (12000000, 125000), (12000000, 5000), (12000000, 1000), (12000000, 100), (12000000, 10), (12000000, 0), (6000000, 0), (3000000, 0), (1000000, 0)]
		'''
		rates = self._cmd('GET_ALL_SAMPLE_RATES')
		self.sample_rates = []
		for line in rates.split('\n'):
			if len(line):
				digital, analog = list(map(int, map(str.strip, line.split(','))))
				self.sample_rates.append((digital, analog))
		return self.sample_rates

	def get_bandwidth(self, sample_rate, device = None, channels = None):
		'''Compute USB bandwidth for a given configuration.

		Must supply sample_rate because Saleae API has no get_sample_rate method.

		>>> s.get_bandwidth(s.get_all_sample_rates()[0])
		96000000
		'''
		# From https://github.com/ppannuto/python-saleae/issues/8
		# Bandwidth (bits per second) =
		#   (digital_sample_rate * digital_channel_count) +
		#   (analog_sample_rate * analog_channel_count * adc_width)
		#
		# ADC width = 12 bits for Logic 8, Pro 8 and Pro 16.
		# ADC width = 8 bits for logic 4.
		if device is None:
			device = self.get_active_device()
		if channels is None:
			digital_channels, analog_channels = self.get_active_channels()
		else:
			digital_channels, analog_channels = channels

		if device.type == 'LOGIC_4_DEVICE':
			adc_width = 8
		else:
			adc_width = 12

		return sample_rate[0] * len(digital_channels) +\
		       sample_rate[1] * len(analog_channels) * adc_width

	def get_performance(self):
		'''Get performance value. Performance controls USB traffic and quality.

		:returns: A ``saleae.PerformanceOption``

		>>> s.get_performance() #doctest:+SKIP
		<PerformanceOption.Full: 100>
		'''
		try:
			return PerformanceOption(int(self._cmd("GET_PERFORMANCE")))
		except self.CommandNAKedError:
			log.warn("get_performance is only supported when a physical Saleae device is attached if")
			log.warn("                you are testing / do not have a Saleae attached this will fail.")
			raise

	def set_performance(self, performance):
		'''Set performance value. Performance controls USB traffic and quality.

		:param performance: must be of type saleae.PerformanceOption

		**Note: This will change the sample rate.**

		#>>> s.set_performance(saleae.PerformanceOption.Full)
		'''
		# Ensure this is a valid setting
		performance = PerformanceOption(performance)
		try:
			self._cmd('SET_PERFORMANCE, {}'.format(performance.value))
		except self.CommandNAKedError:
			log.warn("set_performance is only supported when a physical Saleae device is attached if")
			log.warn("                you are testing / do not have a Saleae attached this will fail.")
			raise

	def get_capture_pretrigger_buffer_size(self):
		'''The number of samples saleae records before the trigger.

		:returns: An integer number descripting the pretrigger buffer size

		>>> s.get_capture_pretrigger_buffer_size() #doctest:+ELLIPSIS
		1...
		'''
		return int(self._cmd('GET_CAPTURE_PRETRIGGER_BUFFER_SIZE'))

	def set_capture_pretrigger_buffer_size(self, size, round=True):
		'''Set the number of samples saleae records before the trigger.

		>>> s.set_capture_pretrigger_buffer_size(1e6)
		'''
		valid_sizes = (1000000, 10000000, 100000000, 1000000000)
		if round:
			size = self._round_up_or_max(size, valid_sizes)
		elif size not in valid_sizes:
			raise NotImplementedError("Invalid size")
		self._cmd('SET_CAPTURE_PRETRIGGER_BUFFER_SIZE, {}'.format(size))

	def get_connected_devices(self):
		'''Get a list of attached Saleae devices.

		Note, this will never be an empty list. If no actual Saleae devices are
		connected, then Logic will return the four fake devices shown in the
		example.

		:returns: A list of ``saleae.ConnectedDevice`` objects

		>>> s.get_connected_devices() #doctest:+ELLIPSIS
		[<saleae.ConnectedDevice #1 LOGIC_4_DEVICE Logic 4 (...) **ACTIVE**>, <saleae.ConnectedDevice #2 LOGIC_8_DEVICE Logic 8 (...)>, <saleae.ConnectedDevice #3 LOGIC_PRO_8_DEVICE Logic Pro 8 (...)>, <saleae.ConnectedDevice #4 LOGIC_PRO_16_DEVICE Logic Pro 16 (...)>]
		'''
		devices = self._cmd('GET_CONNECTED_DEVICES')
		# command response is sometimes not the expected one : a non-empty string starting with a digit (index)
		while ('' == devices or not devices[0].isdigit()):
			time.sleep(0.1)
			devices = self._cmd('GET_CONNECTED_DEVICES')

		self.connected_devices = []
		for dev in devices.split('\n')[:-1]:
			active = False
			try:
				index, name, type, id, active = list(map(str.strip, dev.split(',')))
			except ValueError:
				index, name, type, id = list(map(str.strip, dev.split(',')))
			self.connected_devices.append(ConnectedDevice(type, name, id, index, active))
		return self.connected_devices

	def get_active_device(self):
		'''Get the current active Saleae device.

		:returns: A ``saleae.ConnectedDevice`` object for the active Saleae

		>>> s.get_active_device() #doctest:+ELLIPSIS
		<saleae.ConnectedDevice #1 LOGIC_4_DEVICE Logic 4 (...) **ACTIVE**>
		'''
		self.get_connected_devices()
		for dev in self.connected_devices:
			if dev.active:
				return dev
		raise NotImplementedError("No active device?")

	def select_active_device(self, device_index):
		'''
		>>> s.select_active_device(2)
		>>> s.get_active_device() #doctest:+ELLIPSIS
		<saleae.ConnectedDevice #2 LOGIC_8_DEVICE Logic 8 (...) **ACTIVE**>
		>>> s.select_active_device(1)
		'''
		if self.connected_devices is None:
			self.get_connected_devices()
		for dev in self.connected_devices:
			if dev.index == device_index:
				self._cmd('SELECT_ACTIVE_DEVICE, {}'.format(device_index))
				break
		else:
			raise NotImplementedError("Device index not in connected_devices")

	def get_active_channels(self):
		'''Get the active digital and analog channels.

		:returns: A 2-tuple of lists of integers, the active digital and analog channels respectively

		>>> s.get_active_channels()
		([0, 1, 2, 3], [0])
		'''
		# If an old Logic8 is connected this command does not work, but all 8
		# digital channels are always active so return that.
		device = self.get_active_device()
		if device.type == "LOGIC_DEVICE":
			return range(8), []

		channels = self._cmd('GET_ACTIVE_CHANNELS')
		# Work around possible bug in Logic8
		# https://github.com/ppannuto/python-saleae/pull/19
		while not channels.startswith('digital_channels'):
			time.sleep(0.1)
			channels = self._cmd('GET_ACTIVE_CHANNELS')
		msg = list(map(str.strip, channels.split(',')))
		assert msg.pop(0) == 'digital_channels'
		i = msg.index('analog_channels')
		digital = list(map(int, msg[:i]))
		analog = list(map(int, msg[i+1:]))

		return digital, analog

	def set_active_channels(self, digital=None, analog=None):
		'''Set the active digital and analog channels.

		*Note: This feature is only supported on Logic 16, Logic 8(2nd gen),
		Logic Pro 8, and Logic Pro 16*

		:raises ImpossibleSettings: if used with a Logic 4 device
		:raises ImpossibleSettings: if no active channels are given

		>>> s.set_active_channels([0,1,2,3], [0]) #doctest:+SKIP
		'''
		# Logic 4 doesn't support setting channels over the scripting server:
		# https://github.com/saleae/SaleaeSocketApi/blob/master/SaleaeSocketApi/SocketApi.cs#L899
		if self.get_active_device().type == 'LOGIC_4_DEVICE':
			raise self.ImpossibleSettings("Logic 4 does not support setting channels")

		# TODO Enfore note from docstring
		digital_no = 0 if digital is None else len(digital)
		analog_no = 0 if analog is None else len(analog)
		if digital_no <= 0 and analog_no <= 0:
			raise self.ImpossibleSettings('Logic requires at least one activate channel (digital or analog) and none are given')

		self._build('SET_ACTIVE_CHANNELS')
		if digital_no > 0:
			self._build('digital_channels')
			self._build(['{0:d}'.format(ch) for ch in digital])
		if analog_no > 0:
			self._build('analog_channels')
			self._build(['{0:d}'.format(ch) for ch in analog])
		self._finish()

	def reset_active_channels(self):
		'''Set all channels to active.

		>>> s.reset_active_channels()
		'''
		self._cmd('RESET_ACTIVE_CHANNELS')

	def capture_start(self):
		'''Start a new capture and immediately return.'''
		self._cmd('CAPTURE', False)

	def capture_start_and_wait_until_finished(self):
		'''Convenience method that blocks until capture is complete.

		>>> s.set_capture_seconds(.5)
		>>> s.capture_start_and_wait_until_finished()
		>>> s.is_processing_complete()
		True
		'''
		self.capture_start()
		while not self.is_processing_complete():
			time.sleep(0.1)

	def capture_stop(self):
		'''Stop a capture and return whether any data was captured.

		:returns: True if any data collected, False otherwise

		>>> s.set_capture_seconds(5)
		>>> s.capture_start()
		>>> time.sleep(1)
		>>> s.capture_stop()
		True
		'''
		try:
			self._cmd('STOP_CAPTURE')
			return True
		except self.CommandNAKedError:
			return False

	def capture_to_file(self, file_path_on_target_machine):
		if os.path.splitext(file_path_on_target_machine)[1] == '':
			file_path_on_target_machine += '.logicdata'
		# Fix windows path if needed
		file_path_on_target_machine.replace('\\', '/')
		self._cmd('CAPTURE_TO_FILE, ' + file_path_on_target_machine)

	def get_inputs(self):
		raise NotImplementedError("Saleae temporarily dropped this command")

	def is_processing_complete(self):
		resp = self._cmd('IS_PROCESSING_COMPLETE', expect_nak=True)
		if resp is None:
			return False
		return resp.strip().upper() == 'TRUE'

	def save_to_file(self, file_path_on_target_machine):
		while not self.is_processing_complete():
			time.sleep(1)
		# Fix windows path if needed
		file_path_on_target_machine.replace('\\', '/')
		self._cmd('SAVE_TO_FILE, ' + file_path_on_target_machine)

	def load_from_file(self, file_path_on_target_machine):
		# Fix windows path if needed
		file_path_on_target_machine.replace('\\', '/')
		self._cmd('LOAD_FROM_FILE, ' + file_path_on_target_machine)

	def close_all_tabs(self):
		self._cmd('CLOSE_ALL_TABS')

	def export_data(self,
			file_path_on_target_machine,
			digital_channels=None,
			analog_channels=None,
			analog_format="voltage",
			time_span=None,				# 'None-->all_time, [x.x, y.y]-->time_span'
			format="csv",				# 'csv, bin, vcd, matlab'
			csv_column_headers=True,
			csv_delimeter='comma',		# 'comma' or 'tab'
			csv_timestamp='time_stamp',	# 'time_stamp, sample_number'
			csv_combined=True,			# 'combined' else 'separate'
			csv_row_per_change=True,	# 'row_per_change' else 'row_per_sample'
			csv_number_format='hex',	# dec, hex, bin, ascii
			bin_per_change=True,		# 'on_change' else 'each_sample'
			bin_word_size='8'			# 8, 16, 32, 64
			):
		# export_data, C:\temp_file, digital_channels, 0, 1, analog_channels, 1, voltage, all_time, adc, csv, headers, comma, time_stamp, separate, row_per_change, Dec
		# export_data, C:\temp_file, all_channels, time_span, 0.2, 0.4, vcd
		# export_data, C:\temp_file, analog_channels, 0, 1, 2, adc, all_time, matlab

		frame = inspect.currentframe().f_back
		warnings.warn_explicit('export_data is deprecated, use export_data2',
				category=UserWarning, # DeprecationWarning suppressed by default
				filename=inspect.getfile(frame.f_code),
				lineno=frame.f_lineno)

		while not self.is_processing_complete():
			time.sleep(1)

		# The path needs to be absolute. This is hard to check reliably since we
		# don't know the OS on the target machine, but we can do a basic check
		# for something that will definitely fail
		if file_path_on_target_machine[0] in ('~', '.'):
			raise NotImplementedError('File path must be absolute')
		# Fix windows path if needed
		file_path_on_target_machine.replace('\\', '/')
		self._build('EXPORT_DATA')
		self._build(file_path_on_target_machine)
		if (digital_channels is None) and (analog_channels is None):
			self._build('all_channels')
			analog_channels = self.get_active_channels()[1]
		else:
			if digital_channels is not None and len(digital_channels):
				self._build('digital_channels')
				for ch in digital_channels:
					self._build(str(ch))
			if analog_channels is not None and len(analog_channels):
				self._build('analog_channels')
				for ch in analog_channels:
					self._build(str(ch))
		if analog_channels is not None and len(analog_channels):
			if analog_format not in ('voltage', 'adc'):
				raise NotImplementedError("bad analog_format")
			self._build(analog_format)

		if time_span is None:
			self._build('all_time')
		elif len(time_span) == 2:
			self._build('time_span')
			self._build(str(time_span[0]))
			self._build(str(time_span[1]))
		else:
			raise NotImplementedError('invalid time format')

		if format == 'csv':
			self._build(format)

			if csv_column_headers:
				self._build('headers')
			else:
				self._build('no_headers')

			if csv_delimeter not in ('comma', 'tab'):
				raise NotImplementedError('bad csv delimeter')
			self._build(csv_delimeter)

			if csv_timestamp not in ('time_stamp', 'sample_number'):
				raise NotImplementedError('bad csv timestamp')
			self._build(csv_timestamp)

			if csv_combined:
				self._build('combined')
			else:
				self._build('separate')

			if csv_row_per_change:
				self._build('row_per_change')
			else:
				self._build('row_per_sample')

			if csv_number_format not in ('dec', 'hex', 'bin', 'ascii'):
				raise NotImplementedError('bad csv number format')
			self._build(csv_number_format)
		elif format == 'bin':
			self._build(format)

			if bin_per_change:
				self._build('on_change')
			else:
				self._build('each_sample')

			if bin_word_size not in ('8', '16', '32', '64'):
				raise NotImplementedError('bad bin word size')
			self._build(bin_word_size)

		elif format in ('vcd', 'matlab'):
			# No options for these
			self._build(format)
		else:
			raise NotImplementedError('unknown format')

		self._finish()


	def _export_data2_analog_binary(self, analog_format='voltage'):
		'''Binary analog: [VOLTAGE|ADC]'''

		# Do argument verification
		if analog_format.lower() not in ['voltage', 'adc']:
			raise self.ImpossibleSettings('Unsupported binary analog format')

		# Build arguments
		self._build(analog_format.upper())

	# NOTE: the [EACH_SAMPLE|ON_CHANGE] is the same as the CSV [ROW_PER_CHANGE|ROW_PER_SAMPLE], but I am using name convention from official C# API
	def _export_data2_digital_binary(self, each_sample=True, no_shift=True, word_size=16):
		'''Binary digital: [EACH_SAMPLE|ON_CHANGE], [NO_SHIFT|RIGHT_SHIFT], [8|16|32|64]'''
		# Do argument verification
		if word_size not in [8, 16, 32, 64]:
			raise self.ImpossibleSettings('Unsupported binary word size')

		# Build arguments
		self._build('EACH_SAMPLE' if each_sample else 'ON_CHANGE')
		self._build('NO_SHIFT' if no_shift else 'RIGHT_SHIFT')
		self._build(str(word_size))

	def _export_data2_analog_csv(self, column_headers=True, delimiter='comma', display_base='hex', analog_format='voltage'):
		'''CVS export analog/mixed: [HEADERS|NO_HEADERS], [COMMA|TAB], [BIN|DEC|HEX|ASCII], [VOLTAGE|ADC]'''

		# Do argument verification
		if delimiter.lower() not in ['comma', 'tab']:
			raise self.ImpossibleSettings('Unsupported CSV delimiter')
		if display_base.lower() not in ['bin', 'dec', 'hex', 'ascii']:
			raise self.ImpossibleSettings('Unsupported CSV display base')
		if analog_format.lower() not in ['voltage', 'adc']:
			raise self.ImpossibleSettings('Unsupported CSV analog format')

		# Build arguments
		self._build('HEADERS' if column_headers else 'NO_HEADERS')
		self._build(delimiter.upper())
		self._build(display_base.upper())
		self._build(analog_format.upper())

	def _export_data2_digital_csv(self, column_headers=True, delimiter='comma', timestamp='time_stamp', display_base='hex', rows_per_change=True):
		'''CVS export digital: [HEADERS|NO_HEADERS], [COMMA|TAB], [TIME_STAMP|SAMPLE_NUMBER], [COMBINED, [BIN|DEC|HEX|ASCII]|SEPARATE], [ROW_PER_CHANGE|ROW_PER_SAMPLE]'''

		# Do argument verification
		if delimiter.lower() not in ['comma', 'tab']:
			raise self.ImpossibleSettings('Unsupported CSV delimiter')
		if timestamp.lower() not in ['time_stamp', 'sample_number']:
			raise self.ImpossibleSettings('Unsupported timestamp setting')
		if display_base.lower() not in ['bin', 'dec', 'hex', 'ascii', 'separate']:
			raise self.ImpossibleSettings('Unsupported CSV display base')

		# Build arguments
		self._build('HEADERS' if column_headers else 'NO_HEADERS')
		self._build(delimiter.upper())
		self._build(timestamp.upper())
		self._build('SEPARATE' if display_base.upper() == 'SEPARATE' else ['COMBINED', display_base.upper()])
		self._build('ROW_PER_CHANGE' if rows_per_change else 'ROW_PER_SAMPLE')

	def _export_data2_digital_vcd(self):
		'''VCD digital: no arguments'''
		pass

	def _export_data2_analog_matlab(self, analog_format='voltage'):
		'''Matlab analog: [VOLTAGE|ADC]'''

		# Do argument verification
		if analog_format.lower() not in ['voltage', 'adc']:
			raise self.ImpossibleSettings('Unsupported Matlab analog format')

		# Build arguments
		self._build(analog_format.upper())

	def _export_data2_digital_matlab(self):
		'''Matlab digital: no arguments'''
		pass


	def export_data2(self, file_path_on_target_machine, digital_channels=None, analog_channels=None, time_span=None, format='csv', **export_args):
		'''Export command:
			EXPORT_DATA2,
			<filename>,
			[ALL_CHANNELS|SPECIFIC_CHANNELS, [DIGITAL_ONLY|ANALOG_ONLY|ANALOG_AND_DIGITAL], <channel index> [ANALOG|DIGITAL], ..., <channel index> [ANALOG|DIGITAL]],
			[ALL_TIME|TIME_SPAN, <(double)start>, <(double)end>],
			[BINARY, <binary settings>|CSV, <csv settings>|VCD|MATLAB, <matlab settings>]

		>>> s.export_data2('/tmp/test.csv')
		'''
		while not self.is_processing_complete():
			time.sleep(1)

		# NOTE: Note to Saleae, Logic should resolve relative paths, I do not see reasons not to do this ...
		if file_path_on_target_machine[0] in ('~', '.'):
			raise ValueError('File path must be absolute')
		# Fix windows path if needed
		file_path_on_target_machine.replace('\\', '/')
		
		#Get active channels
		digital_active, analog_active = self.get_active_channels()
		
		self._build('EXPORT_DATA2')
		self._build(file_path_on_target_machine)

		# Channel selection
		is_analog = False
		if (digital_channels is None) and (analog_channels is None):
			self._build('ALL_CHANNELS')
			is_analog = len(self.get_active_channels()[1]) > 0
		else:
			self._build('SPECIFIC_CHANNELS')
			# Check for mixed mode
			# NOTE: This feels redundant, we can see if digital only, analog
			# only or mixed from parsing the channels right?!  especially given
			# the fact that only ANALOG_AND_DIGITAL is printed and never
			# DIGITAL_ONLY or ANALOG_ONLY (according to Saleae C#
			# implementation)
			if len(digital_active) and len(analog_active):
				if digital_channels is not None and len(digital_channels) and analog_channels is not None and len(analog_channels):
					self._build('ANALOG_AND_DIGITAL')
				elif digital_channels is not None and len(digital_channels):
					self._build('DIGITAL_ONLY')
				elif analog_channels is not None and len(analog_channels):
					self._build('ANALOG_ONLY')

			# Add in the channels
			if digital_channels is not None and len(digital_channels):
				self._build(['{0:d} DIGITAL'.format(ch) for ch in digital_channels])
			if analog_channels is not None and len(analog_channels):
				self._build(['{0:d} ANALOG'.format(ch) for ch in analog_channels])
				is_analog = True

		# Time selection
		if time_span is None:
			self._build('ALL_TIME')
		elif len(time_span) == 2:
			self._build(['TIME_SPAN', '{0:f}'.format(time_span[0]), '{0:f}'.format(time_span[0])])
		else:
			raise self.ImpossibleSettings('Unsupported time span')

		# Find exporter
		export_name = '_export_data2_{0:s}_{1:s}'.format('analog' if is_analog else 'digital', format.lower())
		if not hasattr(self, export_name):
			raise NotImplementedError('Unsupported export format given ({0:s})'.format(export_name))

		# Let specific export function handle arguments
		self._build(format.upper())
		getattr(self, export_name)(**export_args)

		self._finish()
		time.sleep(0.050) # HACK: Delete me when Logic (saleae) race conditions are fixed


	def get_analyzers(self):
		'''Return a list of analyzers currently in use, with indexes.'''
		reply = self._cmd('GET_ANALYZERS')
		self.analyzers = []
		for line in reply.split('\n'):
			if len(line):
				analyzer_name = line.split(',')[0]
				analyzer_index = int(line.split(',')[1])
				self.analyzers.append((analyzer_name, analyzer_index))
		return self.analyzers

	def export_analyzer(self, analyzer_index, save_path, wait_for_processing=True, data_response=False):
		'''Export analyzer index N and save to absolute path save_path. The analyzer must be finished processing'''
		if wait_for_processing:
			while not self.is_analyzer_complete(analyzer_index):
				time.sleep(0.1)
		self._build('EXPORT_ANALYZER')
		self._build(str(analyzer_index))
		self._build(save_path)
		if data_response:
			self._build('data_response')  # any old extra parameter can be used
		resp = self._finish()
		return resp if data_response else None

	def is_analyzer_complete(self, analyzer_index):
		'''check to see if analyzer with index N has finished processing.'''
		self._build('IS_ANALYZER_COMPLETE')
		self._build(str(analyzer_index))
		resp = self._finish()
		return resp.strip().upper() == 'TRUE'


def demo(host='localhost', port=10429):
	'''A demonstration / usage guide that mirrors Saleae's C# demo'''

	print("Running Saleae connection demo.\n")

	s = Saleae(host=host, port=port)
	print("Saleae connected.")
	input("Press Enter to continue...\n")

	try:
		s.set_performance(PerformanceOption.Full)
		print("Set performance to full.")
	except s.CommandNAKedError:
		print("Could not set performance.")
		print("\tIs a physical Saleae device connected? This command only works")
		print("\twhen actual hardware is plugged in. You can skip it if you are")
		print("\tjust trying things out.")
	input("Press Enter to continue...\n")

	devices = s.get_connected_devices()
	print("Connected devices:")
	for device in devices:
		print("\t{}".format(device))

	# n.b. there are always a few connected test devices if no real HW
	if len(devices) > 1:
		i = int(input("Choose active device (collect data from which Saleae?) [1-{}] ".format(len(devices))))
		while i < 1 or i > len(devices):
			print("You must select a valid device index")
			i = int(input("Choose active device (collect data from which Saleae?) [1-{}] ".format(len(devices))))

		s.select_active_device(i)
		print("Connected devices:")
		devices = s.get_connected_devices()
		for device in devices:
			print("\t{}".format(device))
	else:
		print("Only one Saleae device. Skipping device selection")
	input("Press Enter to continue...\n")

	if s.get_active_device().type == 'LOGIC_4_DEVICE':
		print("Logic 4 does not support setting active channels; skipping")
	else:
		digital = [0,1,2,3,4]
		analog = [0,1]
		print("Setting active channels (digital={}, analog={})".format(digital, analog))
		s.set_active_channels(digital, analog)
		input("Press Enter to continue...\n")

	digital, analog = s.get_active_channels()
	print("Reading back active channels:")
	print("\tdigital={}\n\tanalog={}".format(digital, analog))
	input("Press Enter to continue...\n")

	print("Setting to capture 2e6 samples")
	s.set_num_samples(2e6)
	input("Press Enter to continue...\n")

	print("Setting to sample rate to at least digitial 4 MS/s, analog 100 S/s")
	rate = s.set_sample_rate_by_minimum(4e6, 100)
	print("\tSet to", rate)
	input("Press Enter to continue...\n")

	print("Starting a capture")
	# Also consider capture_start_and_wait_until_finished for non-demo apps
	s.capture_start()
	while not s.is_processing_complete():
		print("\t..waiting for capture to complete")
		time.sleep(1)
	print("Capture complete")

	print("")
	print("Demo complete.")


if __name__ == '__main__':
	demo()



## Support bits for doctests:

# n.b. DocTestRunner is an old-style class so no super for py2k compat
import doctest
_original_runner = doctest.DocTestRunner
class CustomRunner(_original_runner):
	def __init__(self, *args, **kwargs):
		_original_runner.__init__(self, *args, **kwargs)

		self._saleae = Saleae()

	def run(self, test, *args, **kwargs):
		test.globs['s'] = self._saleae
		return _original_runner.run(self, test, *args, **kwargs)

def setup_module(module):
	doctest.DocTestRunner = CustomRunner

