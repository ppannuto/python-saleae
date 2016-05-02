#!/usr/bin/env python3

# One common issue is that Saleae records traces into memory, which means that
# it can't handle very long captures. This example shows how to use scripting to
# do long recordings over time. There will be brief gaps every time Saleae saves
# the old recording and starts a new one.

import os
import time

import saleae

folder = time.strftime('%Y-%m-%d--%H-%M-%S')
os.mkdir(folder)

s = saleae.Saleae()

# Note: This is a short number of samples. You'll probably want more.
s.set_num_samples(1e6)

for i in range(5):
	path = os.path.abspath(os.path.join(folder, str(i)))
	s.capture_to_file(path)

