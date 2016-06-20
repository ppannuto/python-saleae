from nose.tools import *
import saleae

class TestSaleae(object):
	@classmethod
	def setup_class(cls):
		cls.s = saleae.Saleae()

	@classmethod
	def teardown(cls):
		pass

# Moved tests over to doctests as they seem to work well for current needs.
# This file is left as a stub in case more complex tests are ever warranted.

