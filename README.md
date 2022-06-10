# University of Economics in Katowice - Computational Intelligence course project

## Overview
This project is done for the Computational Intelligence course @ University of Economics in Katowice CS Masters program. The purpose of this project is to write a program that solves a complex Vechicle Routing Problem task. Genetic algorithm approach has been chosen as the solver for the problem.

## Included constraints/requirements:
✅ Points generation and visualization.\
✅ Cars creation.\
✅ Draw of amount of commodities for supply/demand customers.\
✅ Different products: Tuna, Oranges, Uran.\
✅ Easter egg - cat driving and eating Tuna in cars.\
✅ Working optimization with Genetic Algorithm *(sic!)*.\
✅ Routes visualization.\
✅ High-level pseudocode of the solution.\
✅ Proper docstrings in the code.\
✅ Project manager photo 🐈😻 (see below...😉)\
✅ **CLEAN CODE** 👼

## Overall pseudocode of the solution
```
1. Generate random points.
2. Draw initial magazines for each of the cars.
3. Draw the cat to one of 6 cars.
4. Learning process:
    4.1 Create base population.
    4.2 Evaluate base population.
    
    4.3 while iteration != n_iterations:
        4.3.1 iteration += 1
        4.3.2 Perform selection.
        4.3.3 Perform mutation.
        4.3.4 Perform crossover.
        4.3.5 Evaluate new population.
    
    4.4 Select best score and routes.
```

## Authors
- Szymon Stolarski
- Katarzyna Majchrzak
- Przemysław Bentkowski
- Piotr Jonderko
- Shiro - Project Manager

<img src="components/pictures/shiro_the_cat.png" width="500" height=700>

## Bugs 🐞
🤦‍♂️ The cars stopped driving from/through magazines