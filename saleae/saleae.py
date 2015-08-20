#!/usr/bin/env python3

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import bisect
import enum
import os
import socket
import sys
import time

class SaleaeConnection():
	class SaleaeConnectionError(Exception):
		pass

	class CommandNAKedError(SaleaeConnectionError):
		pass

	def __init__(self, host='localhost', port=10429):
		self._to_send = []
		self.sample_rates = None
		self.connected_devices = None

		self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._s.connect((host, port))
		self._rxbuf = ''

	def _build(self, s):
		'''Convenience method for building up a command to send'''
		self._to_send.append(s)

	def _abort(self):
		self._to_send = []

	def _finish(self, s=None):
		if s:
			self._build(s)
		self._send(', '.join(self._to_send))

	def _round_up_or_max(self, value, candidates):
		i = bisect.bisect_left(value, candidates)
		if i == len(candidates):
			i -= 1
		return candidates[i]

	def _send(self, s):
		log.debug("Send >{}<".format(s))
		self._s.send(bytes(s + '\0', 'UTF-8'))

	def _recv(self):
		while 'ACK' not in self._rxbuf:
			self._rxbuf += self._s.recv(1024).decode('UTF-8')
			log.debug("Recv >{}<".format(self._rxbuf))
			if 'NAK' == self._rxbuf[0:3]:
				raise self.CommandNAKedError
		ret, self._rxbuf = self._rxbuf.split('ACK', 1)
		return ret

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

	def set_trigger(self, channels):
		self._build('SET_TRIGGER')
		for c in channels:
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

	def set_num_samples(self, samples):
		self._send('SET_NUM_SAMPLES, {}'.format(samples))

	def set_capture_seconds(self, seconds):
		self._send('SET_CAPTURE_SECONDS, {}'.format(seconds))

	def set_sample_rate(self, digital_rate, analog_rate, round=True):
		'''Only fixed sample rates are supported.'''

		if self.sample_rates is None:
			self.get_all_sample_rates()
		if (digital_rate, analog_rate) not in self.sample_rates:
			raise NotImplementedError("Unsupported sample rate")

		self._send('SET_SAMPLE_RATE, {}, {}'.format(digital_rate, analog_rate))

	def get_all_sample_rates(self):
		self._send('GET_ALL_SAMPLE_RATES')
		rates = self._recv()
		self.sample_rates = []
		for line in rates.split('\n'):
			if len(line):
				digital, analog = list(map(int, map(str.strip, line.split(','))))
				self.sample_rates.append((digital, analog))
		return self.sample_rates

	def get_performance(self):
		self._send("GET_PERFORMANCE")
		return self.PerformanceOption(self._recv())

	def set_performance(self, performance):
		# Ensure this is a valid setting
		performance = self.PerformanceOption(performance)
		self._send('SET_PERFORMANCE_OPTION, {}'.format(performance.value))

	def get_capture_pretrigger_buffer_size(self):
		self._send('GET_CAPTURE_PRETRIGGER_BUFFER_SIZE')
		return self._recv()

	def set_capture_pretrigger_buffer_size(self, size, round=True):
		valid_sizes = (1000000, 10000000, 100000000, 1000000000)
		if round:
			size = self._round_up_or_max(size, valid_sizes)
		elif size not in valid_sizes:
			raise NotImplementedError("Invalid size")
		self._send('SET_CAPTURE_PRETRIGGER_BUFFER_SIZE, {}'.format(size))

	class ConnectedDevice():
		def __init__(self, type, name, id, index, active):
			self.type = type
			self.name = name
			self.id = int(id, 16)
			self.index = int(index)
			self.active = bool(active)

		def __str__(self):
			if self.active:
				return "<saleae.ConnectedDevice #{self.index} {self.name} ({self.id:x}) {self.type} ACTIVE>".format(self=self)
			else:
				return "<saleae.ConnectedDevice #{self.index} {self.name} ({self.id:x}) {self.type}>".format(self=self)

		def __repr__(self):
			return str(self)

	def get_connected_devices(self):
		self._send('GET_CONNECTED_DEVICES')
		self.connected_devices = []
		for dev in self._recv().split('\n')[:-1]:
			active = False
			try:
				index, name, type, id, active = list(map(str.strip, dev.split(',')))
			except ValueError:
				index, name, type, id = list(map(str.strip, dev.split(',')))
			self.connected_devices.append(self.ConnectedDevice(type, name, id, index, active))
		return self.connected_devices

	def select_active_device(self, device_index):
		if self.connected_devices is None:
			self.get_connected_devices()
		for dev in self.connected_devices:
			if dev.index == device_index:
				self._send('SELECT_ACTIVE_DEVICE, {}'.format(device_index))
				break
		else:
			raise NotImplementedError("Device index not in connected_devices")

	def get_active_channels(self):
		self._send('GET_ACTIVE_CHANNELS')
		msg = list(map(str.strip, self._recv().split(',')))
		assert msg.pop(0) == 'digital_channels'
		i = msg.index('analog_channels')
		digital = list(map(int, msg[:i]))
		analog = list(map(int, msg[i+1:]))

		return digital, analog

	def set_active_channels(self, digital, analog):
		# TODO Enforce "Note: This feature is only supported on Logic 16,
		# Logic 8(2nd gen), Logic Pro 8, and Logic Pro 16"
		self._build('SET_ACTIVE_CHANNELS digital_channels')
		for ch in digital:
			self._build('{}'.format(ch))
		self._build(', analog_channels')
		for ch in analog:
			self._build('{}'.format(ch))
		self._finish()

	def reset_active_channels(self):
		'''Sets all channels to active'''
		self._send('RESET_ACTIVE_CHANNELS')

	def capture(self):
		self._send('CAPTURE')

	def stop_capture(self):
		self._send('STOP_CAPTURE')
		self._recv()
		raise NotImplementedError

	def capture_to_file(self, file_path_on_target_machine):
		if os.path.splitext(file_path_on_target_machine)[1] == '':
			file_path_on_target_machine += '.logicdata'
		self._send('CAPTURE_TO_FILE, ' + file_path_on_target_machine)

	def get_inputs(self):
		raise NotImplementedError("Saleae temporarily dropped this command")

	def is_processing_complete(self):
		raise NotImplementedError

	def save_to_file(self, file_path_on_target_machine):
		while not self.is_processing_complete():
			time.sleep(1)
		self._send('SAVE_TO_FILE, ' + file_path_on_target_machine)

	def load_from_file(self, file_path_on_target_machine):
		self._send('LOAD_TO_FILE, ' + file_path_on_target_machine)

	def close_all_tabs(self):
		self._send('CLOSE_ALL_TABS')

	def export_data(self,
			file_path_on_target_machine,
			digital_channels=None,
			analog_channels=None,
			analog_format="voltage",
			time="all_time",
			format="csv",				# 'csv, bin, vcd, matlab'
			csv_column_headers=True,
			csv_delimeter='comma',		# 'comma' or 'tab'
			csv_number_format='hex',	# dec, hex, bin, ascii
			):
		# export_data, C:\temp_file, digital_channels, 0, 1, analog_channels, 1, voltage, all_time, adc, csv, headers, comma, time_stamp, separate, row_per_change, Dec
		# export_data, C:\temp_file, all_channels, time_span, 0.2, 0.4, vcd
		# export_data, C:\temp_file, analog_channels, 0, 1, 2, adc, all_time, matlab

		while not self.is_processing_complete():
			time.sleep(1)

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

		if time not in ('all_time',):
			raise NotImplementedError('times other that all_time')
		self._build('all_time')

		if format == 'csv':
			self._build('csv')

			if csv_column_headers:
				self._build('headers')
			else:
				self._build('no_headers')

			if csv_delimeter not in ('comma', 'tab'):
				raise NotImplementedError('bad csv delimeter')
			self._build(csv_delimeter)

			#TODO the rest of these options
			self._build('time_stamp')
			self._build('combined')

			if csv_number_format not in ('dec', 'hex', 'bin', 'ascii'):
				raise NotImplementedError('bad csv number format')
			self._build(csv_number_format)
		elif format == 'bin':
			raise NotImplementedError('bin format')
		elif format in ('vcd', 'matlab'):
			# No options for these
			self._build(format)
		else:
			raise NotImplementedError('unknown format')

		self._finish()


	def get_analyzers(self):
		raise NotImplementedError
	def export_analyzers(self):
		raise NotImplementedError
	def is_analyzer_complete(self, analyzer_index):
		raise NotImplementedError

