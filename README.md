# AttackBuilder

##  Performance

### PGD

- `step_size=eps/sqrt(num_iteration)`  generates stronger attack than `step_size=eps`.
- `rand_init` sometimes has BAD effect to generate attack.  

| model | defense | norm | num iter | rand init | step size         | eps  | acc  | id |
| ----- | ------- | ---- | -------- | --------  | ----------------- | ---  | ---  | -- |
| resnet56 | -    |      |          |           |                   | 0.0  | 92.9 | clean_model.pth  |
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
|          |      | linf | 20       | True      | eps / sqrt(#iter) | 4.0  | 0.01 |    |
|          |      |      |          |           |                   | 8.0  | 0.0  |    |
|          |      |      |          |           |                   | 16.0 | 0.0  |    |
|          |      |      |          |           | eps               | 4.0  | 0.91 |    |
|          |      |      |          |           |                   | 8.0  | 0.12 |    |
|          |      |      |          |           |                   | 16.0 | 0.05 |    |
|          |      |      |          | False     | eps / sqrt(#iter) | 4.0  | 0.02 |    |
|          |      |      |          |           |                   | 8.0  | 0.0  |    |
|          |      |      |          |           |                   | 16.0 | 0.0  |    |
|          |      |      |          |           | eps               | 4.0  | 0.66 |    |
|          |      |      |          |           |                   | 8.0  | 0.09 |    |
|          |      |      |          |           |                   | 16.0 | 0.06 |    |