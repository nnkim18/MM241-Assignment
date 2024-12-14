# README

## Overview
This project contains two cutting stock problem policies implemented as Python classes. Each policy is designed to handle the allocation of product sizes to available stock grids efficiently. The two classes are:

1. **FirstCutPolicy**: Implements a heuristic-based approach to place products sequentially into stocks by finding valid subgrid positions.
2. **ColumnGeneration**: Uses the column generation technique to iteratively optimize cutting patterns for better resource utilization.

---

## **FirstCutPolicy**

### Description
The `FirstCutPolicy` is a straightforward heuristic-based method that aims to place products into the first available stock where they fit. It prioritizes simplicity and computational efficiency over global optimization.

### Algorithm Steps
1. Iterate through the list of products, checking their demand and sizes.
2. For each product, iterate through all available stocks.
3. Use subgrid checking to find a valid position where the product can fit in the stock.
4. Place the product and update the stock grid.
5. Return the placement action or `None` if no valid placement is found.

## **ColumnGeneration**

### Description
The `ColumnGeneration` class implements the column generation optimization method, a more advanced approach designed to minimize material waste while fulfilling product demand. It combines a master problem and a pricing problem to iteratively refine cutting patterns.

### Algorithm Steps
1. **Initialization**:
   - Generate initial patterns based on product sizes and stock dimensions.
   - Ensure patterns are unique.

2. **Master Problem**:
   - Solve a linear program to allocate demand across existing patterns.
   - Extract dual prices from the solution to guide the pricing problem.

3. **Pricing Problem**:
   - Use dynamic programming to find new patterns with the highest reduced cost.
   - Add the new pattern if it improves the solution.

4. **Pattern Selection**:
   - Choose the pattern that maximizes demand coverage and minimizes cost.

5. **Action Conversion**:
   - Convert the selected pattern into a placement action.

## Comparison of Policies

| Feature                | FirstCutPolicy                    | ColumnGeneration             |
|------------------------|------------------------------------|------------------------------|
| **Type**              | Heuristic                        | Optimization-based           |
| **Efficiency**        | Fast, simpler logic              | Slower, computationally intensive |
| **Optimality**        | Suboptimal, greedy placement     | Near-optimal patterns        |
| **Complexity**        | Low                              | High                        |
| **Use Case**          | Small-scale problems, quick results | Large-scale problems requiring minimal waste |

## Policy Performance Comparison

| Policy              | Filled Ratio | Trim Loss            |
|---------------------|--------------|----------------------|
| **FirstCutPolicy**  | 0.14         | 0.2216599996670104   |
| **ColumnGeneration**| 0.2          | 0.19658451138683425  |