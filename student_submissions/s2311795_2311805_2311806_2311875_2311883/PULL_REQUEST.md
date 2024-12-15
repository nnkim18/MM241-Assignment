# Cutting Stock Problem Implementation

## Overview
Our implementation provides two distinct approaches to solve the cutting stock problem:
1. Policy 1: Best Fit approach with comprehensive position evaluation
2. Policy 2: Column Generation approach with first-fit placement strategy

 

## Implementation Details

### Policy 1: Best Fit
- Core Features:
  - Evaluates all possible positions for optimal placement
  - Considers both waste minimization and space utilization
  - Supports product rotation for better fitting
  - Implements comprehensive position scoring system

 

### Policy 2: Column Generation
- Key Components:
  - Uses first-fit strategy for quick placement
  - Optimizes through column generation approach
  - `_find_position`: Finds first valid position for placement
  - Implements rotation checking for better space utilization

 

## Algorithm Demonstration

### Policy 1: Best Fit Approach
![Best Fit Demo](https://i.imgur.com/YOUR_BESTFIT_GIF_ID.gif)

 

### Policy 2: Column Generation Approach
![Column Generation Demo](https://i.imgur.com/YOUR_COLGEN_GIF_ID.gif)

 

## Results and Evaluation

### Performance Results
#### Best Fit Policy
![Best Fit Results](https://i.imgur.com/YOUR_BESTFIT_RESULT_ID.png)

 

#### Column Generation Policy
![Column Generation Results](https://i.imgur.com/YOUR_COLGEN_RESULT_ID.png)

 

### Advantages
1. Both policies support product rotation
2. Policy 1 minimizes waste through comprehensive position evaluation
3. Policy 2 provides faster placement with reasonable optimization
4. Both approaches handle various product sizes effectively
5. Both policies are designed to be memory-efficient

 

### Disadvantages
1. Policy 1 has higher computational complexity due to exhaustive search
2. Policy 2 might not always find the optimal solution due to first-fit approach

 

## Algorithm Analysis

### Comparison Table
| Evaluate Factor          | Best Fit          | Column Generation | Genetic Algorithm |
|-------------------------|-------------------|-------------------|-------------------|
| Time Complexity         | O(n×m)            | O(n)              | O(g×p×f)          |
| Memory Usage            | Moderate          | Low               | High              |
| Solution Quality        | Good              | Moderate          | Potentially optimal but inconsistent |
| Implementation Complexity| Moderate          | Simple            | Complex           |
| Scalability             | Good for medium-sized problems | Excellent for large problems | Limited by computation time |

 

### Why We Chose Best Fit
After implementing and testing all three approaches, we eventually chose Best Fit instead of Genetic Algorithm because:
1. Easier to implement
2. Unlike GA which can be inconsistent, Best Fit provides reliable results
3. **Faster Execution**: While GA required multiple generations to converge, Best Fit finds good solutions immediately
4. More memory-efficient than GA which needs to maintain a population of solutions

 

### Genetic Algorithm Implementation Notes
Our GA variables:
- Population size: `50` individuals
- Maximum generations: `100`
- Selection method: Tournament selection
- Crossover variants: Uniform and Two-Point
- Mutation rate: `0.1`
- Crossover rate: `0.7`

While the GA showed promise in finding optimal solutions, its computational overhead and complexity was not suitable with our expectations. Therefore, we opted for a simpler approach with a focus on finding good solutions quickly - Best Fit.
