#!/usr/bin/env python3
# vim: tw=80 ts=4 sts=4 sw=4 smarttab noet

import logging
log = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

import bisect
import enum
import os
import socket
import sys
import time

if sys.version_info[0] == 2:
    # if we're running in 2.7 redefine some things
    ConnectionRefusedError = socket.error
    input = raw_input

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

	def __init__(self, host='localhost', port=10429):
		self._to_send = []
		self.sample_rates = None
		self.connected_devices = None

		self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
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
		self._to_send.append(s)

	def _abort(self):
		self._to_send = []

	def _finish(self, s=None):
		if s:
			self._build(s)
		ret = self._cmd(', '.join(self._to_send))
		self._to_send = []
		return ret

	def _round_up_or_max(self, value, candidates):
		i = bisect.bisect_left(value, candidates)
		if i == len(candidates):
			i -= 1
		return candidates[i]

	def _send(self, s):
		log.debug("Send >{}<".format(s))
		# necessary for Python 2.7 compatibility. Since Python 3 moved
		# to all unicode the bytes() function was added and required an
		# `encoding` parameter. The `bytes()` function was backported to
		# Python 2.7 but curiously, doesn't do the same thing. The line in
		# the except block is the functional equivalent for this case
		try:
			byte_packet = bytes(s + '\0', 'UTF-8')
		except TypeError:
			byte_packet = str(bytearray(s + '\0')).encode('UTF-8')

		self._s.send(byte_packet)

	def _recv(self):
		while 'ACK' not in self._rxbuf:
			self._rxbuf += self._s.recv(1024).decode('UTF-8')
			log.debug("Recv >{}<".format(self._rxbuf))
			if 'NAK' == self._rxbuf[0:3]:
				self._rxbuf = self._rxbuf[3:]
				raise self.CommandNAKedError
		ret, self._rxbuf = self._rxbuf.split('ACK', 1)
		return ret

	def _cmd(self, s, wait_for_ack=True):
		self._send(s)

		ret = None
		if wait_for_ack:
			ret = self._recv()
			if sys.version_info[0] == 2:
				ret = ret.encode('UTF-8')
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
		'''
		self._cmd('SET_NUM_SAMPLES, {:d}'.format(int(samples)))

	def set_capture_seconds(self, seconds):
		'''Set the capture duration to a length of time.

		:param seconds: Capture length. Partial seconds (floats) are fine.
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
		'''Get available sample rate combinations for the current performance level and channel combination.'''
		rates = self._cmd('GET_ALL_SAMPLE_RATES')
		self.sample_rates = []
		for line in rates.split('\n'):
			if len(line):
				digital, analog = list(map(int, map(str.strip, line.split(','))))
				self.sample_rates.append((digital, analog))
		return self.sample_rates

	def get_bandwidth(self, sample_rate, device = None, channels = None):
		'''Compute USB bandwidth for a given configuration.

		Must supply sample_rate because Saleae API has no get_sample_rate method.'''
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

		:returns: A ``saleae.PerformanceOption``'''
		return PerformanceOption(int(self._cmd("GET_PERFORMANCE")))

	def set_performance(self, performance):
		'''Set performance value. Performance controls USB traffic and quality.

		:param performance: must be of type saleae.PerformanceOption

		**Note: This will change the sample rate.**'''
		# Ensure this is a valid setting
		performance = PerformanceOption(performance)
		self._cmd('SET_PERFORMANCE, {}'.format(performance.value))

	def get_capture_pretrigger_buffer_size(self):
		return int(self._cmd('GET_CAPTURE_PRETRIGGER_BUFFER_SIZE'))

	def set_capture_pretrigger_buffer_size(self, size, round=True):
		valid_sizes = (1000000, 10000000, 100000000, 1000000000)
		if round:
			size = self._round_up_or_max(size, valid_sizes)
		elif size not in valid_sizes:
			raise NotImplementedError("Invalid size")
		self._cmd('SET_CAPTURE_PRETRIGGER_BUFFER_SIZE, {}'.format(size))

	def get_connected_devices(self):
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
		self.get_connected_devices()
		for dev in self.connected_devices:
			if dev.active:
				return dev
		raise NotImplementedError("No active device?")

	def select_active_device(self, device_index):
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

		:returns: A 2-tuple of lists of integers, the active digital and analog channels respectively'''
		channels = self._cmd('GET_ACTIVE_CHANNELS')
		msg = list(map(str.strip, channels.split(',')))
		assert msg.pop(0) == 'digital_channels'
		i = msg.index('analog_channels')
		digital = list(map(int, msg[:i]))
		analog = list(map(int, msg[i+1:]))

		return digital, analog

	def set_active_channels(self, digital, analog):
		# TODO Enforce "Note: This feature is only supported on Logic 16,
		# Logic 8(2nd gen), Logic Pro 8, and Logic Pro 16"
		self._build('SET_ACTIVE_CHANNELS')
		self._build('digital_channels')
		for ch in digital:
			self._build('{}'.format(ch))
		self._build('analog_channels')
		for ch in analog:
			self._build('{}'.format(ch))
		self._finish()

	def reset_active_channels(self):
		'''Set all channels to active.'''
		self._cmd('RESET_ACTIVE_CHANNELS')

	def capture_start(self):
		'''Start a new capture and immediately return.'''
		self._cmd('CAPTURE', False)

	def capture_start_and_wait_until_finished(self):
		self.capture_start()
		while not self.is_processing_complete():
			time.sleep(0.1)

	def capture_stop(self):
		'''Stop a capture and return whether any data was captured.

		:returns: True if any data collected, False otherwise
		'''
		try:
			self._cmd('STOP_CAPTURE')
			return True
		except self.CommandNAKedError:
			return False

	def capture_to_file(self, file_path_on_target_machine):
		if os.path.splitext(file_path_on_target_machine)[1] == '':
			file_path_on_target_machine += '.logicdata'
		self._cmd('CAPTURE_TO_FILE, ' + file_path_on_target_machine)

	def get_inputs(self):
		raise NotImplementedError("Saleae temporarily dropped this command")

	def is_processing_complete(self):
		resp = self._cmd('IS_PROCESSING_COMPLETE')
		return resp.strip().upper() == 'TRUE'

	def save_to_file(self, file_path_on_target_machine):
		while not self.is_processing_complete():
			time.sleep(1)
		self._cmd('SAVE_TO_FILE, ' + file_path_on_target_machine)

	def load_from_file(self, file_path_on_target_machine):
		self._cmd('LOAD_TO_FILE, ' + file_path_on_target_machine)

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

		while not self.is_processing_complete():
			time.sleep(1)

		# The path needs to be absolute. This is hard to check reliably since we
		# don't know the OS on the target machine, but we can do a basic check
		# for something that will definitely fail
		if file_path_on_target_machine[0] in ('~', '.'):
			raise NotImplementedError('File path must be absolute')

		self._build('EXPORT_DATA')
		self._build(file_path_on_target_machine)
		if (digital_channels is None) and (analog_channels is None):
			self._build('all_channels')
			analog_channels = self.get_active_channels(self)[1]
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

	def export_analyzer(self, analyzer_index, save_path, wait_for_processing=True):
		'''Export analyzer index N and save to absolute path save_path. The analyzer must be finished processing'''
		if wait_for_processing:
			while not self.is_analyzer_complete(analyzer_index):
				time.sleep(0.1)
		self._build('EXPORT_ANALYZER')
		self._build(str(analyzer_index))
		self._build(save_path)
		resp = self._finish()

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
