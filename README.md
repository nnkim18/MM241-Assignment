# Group 141's Submission - Math Modeling Assignment - Semester 241

## Group Members
<!-- Describe cutting stock problem -->
* Nguyễn Phú Thịnh - 2313293
* Đặng Huyền Vũ - 2313950
* Phạm Xuân Bắc - 2310280

## Assignment: Algorithm for Cutting Stock Problem
This pull request introduces the implementation and comparison of two algorithms for solving the **Two-Dimensional Cutting Stock Problem (2D-CSP)**, a critical optimization challenge in manufacturing and logistics. The goal is to minimize material waste while fulfilling specific cutting demands efficiently.

## Algorithms
This implementation features 2 policies aiming to solve the 2D cutting stock problem: the Strip-Packing Algorithm and the Heuristic-Greedy Algorithm. The policy aims to minimize trim loss by considering product orientations and stock utilization.

### Strip-Packing Algorithm (policy_id = 1)


### Heuristic-Greedy Algorithm (policy_id = 2)
This algorithm is the hybrid algorithm which combination of heuristic and greedy. Design for solving 2D-CSP for faster time executed and higher performance. 

## Features
* **Stock Optimization:** Minimizes material waste by optimizing cutting patterns.
* **Algorithm Comparison:** Includes performance analysis of Strip-Packing and Heuristic-Greedy algorithms on trim loss and filled ratio.
* **Scalable Framework:** Supports varying product dimensions and stock sizes.

## Result
| Metrics | Strip-Packing | Heuristic-Greedy |
| ------- | ------------- | ------------- |
| Filled Ratio | 0.18950 | 21.59030  |
| % Trim loss | 0.12580 | 15.80482 |
| Runtime | Slower | Faster |

### Key insights:
* The Strip-Packing algorithm is suitable for high-accuracy needs but slower in execution.
* The Heuristic-Greedy algorithm provides faster runtime and better trim loss reduction, nevertheles it's not stability and only use for a certain testcase.

## Tools and Techniques
* **Programming Language:** Python
* **Library:** NumPy, gymnasium, Scipy
* **Helper Tools:** GitHub, VSCode

