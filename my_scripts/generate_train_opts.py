#!/usr/bin/env python3

import sys, random

for i in range(int(sys.argv[1])):
    print('--lr_{:.5f}_--warmup_{}'.format(float(random.uniform(1e-4, 9e-4)), int(random.uniform(2000, 10000))))
