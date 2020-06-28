# AttackBuilder

##  Performance

### PGD

- `step_size=eps/sqrt(eps)`  is stronger than `step_size=eps`.
- `rand_init` is basically good to genetate strong attack. 

| model | defense | norm | num iter | rand init | step size         | eps  | acc  | id |
| ----- | ------- | ---- | -------- | --------  | ----------------- | ---  | ---  | -- |
| resnet56 | -    |      |          |           |                   | 0.0  | 92.9 | -  |
|          |      | linf | 7        | True      | eps / sqrt(#iter) | 4.0  | 0.26 |    |
|          |      |      |          |           |                   | 8.0  | 0.0  |    |
|          |      |      |          |           |                   | 16.0 | 0.0  |    |
|          |      |      |          |           | eps               | 4.0  | 1.79 |    |
|          |      |      |          |           |                   | 8.0  | 0.30 |    |
|          |      |      |          |           |                   | 16.0 | 0.14 |    |
|          |      |      |          | False     | eps / sqrt(#iter) | 4.0  | 0.24 |    |
|          |      |      |          |           |                   | 8.0  | 0.0  |    |
|          |      |      |          |           |                   | 16.0 | 0.0  |    |
|          |      |      |          |           | eps               | 4.0  | 1.88 |    |
|          |      |      |          |           |                   | 8.0  | 0.40 |    |
|          |      |      |          |           |                   | 16.0 | 0.23 |    |