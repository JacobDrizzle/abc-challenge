# Timing stress: v10_noise_100

Step-20 mean geodesic error (degrees) on the exact grid; the other rows are each model's change.

| Stream | Windows | damped velocity (baseline) | k8 | k8_800 | jitter_1_5 | jitter_3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact 100 Hz grid | 131,459 | 5.0497 | 4.2478 | 4.1812 | 4.2566 | 4.2793 |
| jitter +-1 ms | 131,459 | +0.0010 | +0.0480 | +0.0532 | +0.0168 | +0.0038 |
| jitter +-2 ms | 131,459 | +0.0183 | +0.1573 | +0.1551 | +0.0778 | +0.0277 |
| noise 0.005 deg | 131,459 | +0.0005 | +0.0031 | +0.0041 | +0.0021 | +0.0014 |
| noise 0.01 deg | 131,459 | +0.0019 | +0.0123 | +0.0158 | +0.0083 | +0.0056 |
| noise 0.02 deg | 131,459 | +0.0075 | +0.0458 | +0.0545 | +0.0318 | +0.0218 |
| noise 0.05 deg | 131,459 | +0.0463 | +0.2095 | +0.2121 | +0.1636 | +0.1211 |

Each model minus k8 on the same stream (negative: better than k8).

| Stream | damped velocity (baseline) | k8_800 | jitter_1_5 | jitter_3 |
| --- | ---: | ---: | ---: | ---: |
| exact 100 Hz grid | +0.8019 | -0.0666 | +0.0087 | +0.0315 |
| jitter +-1 ms | +0.7548 | -0.0615 | -0.0225 | -0.0128 |
| jitter +-2 ms | +0.6629 | -0.0689 | -0.0708 | -0.0982 |
| noise 0.005 deg | +0.7993 | -0.0656 | +0.0078 | +0.0298 |
| noise 0.01 deg | +0.7914 | -0.0631 | +0.0048 | +0.0247 |
| noise 0.02 deg | +0.7636 | -0.0579 | -0.0052 | +0.0074 |
| noise 0.05 deg | +0.6386 | -0.0640 | -0.0372 | -0.0569 |
