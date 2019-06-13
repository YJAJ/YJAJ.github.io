---
layout: post
author: YJ Park
title:  "Traveling Salesperson Problem with Genetic Algorithm"
date:   2018-10-14 13:42:59 +1000
categories: jekyll update
tags: Traveling Salesman Problem, TSP, Genetic Algorithm, Python, SQL
---
<head>
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=UA-127453746-1"></script>
    <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'UA-127453746-1');
    </script>
</head>

This assignment, Traveling Salesperson Problem (TSP), consists of the three different parts:
* Part A: Develop a TSP solver;
* Part B: Connect to a database; and
* Part C: Develp a GUI-based TSP solver.

In this blog, I will primarily focus on the development of a TSP solver based on Genetic Algorithm and one of the SQL queries embedded in the solver.

## What is TSP?

The goal of TSP is to identify a minimum cost route where a salesperson is expected to visit every n city only once and return (Deng et al., 2015; Sebő and Vygen, 2012). There are two importance conditions to be noted here: first, a salesperson visits every city once and only once; and second, a salesperson returns to the origin city. 
The first concept on TSP was found in 1759 when Euler was interested in solving the knights’ journey (Jiao and Wang, 2000), followed by a manual originating from 1832 for the scenarios of salesperson travelling 45 German cities without mathematical consideration (Yu et al., 2014). In the 1800s, two mathematicians, Sir William Hamilton and Thomas Kirkman, studied TPS devising mathematical formulations (Matai et al., 2010; Yu et al., 2014). It is, however, noted that the more general form of TSP was initially studied in 1930s by Karl Menger in Vienna and Harvard (Matai et al., 2010). Since then, TSP has fascinated many researchers for several decades as it is a classic non-deterministic polynomial time hard (NP hard) problem.

## Why Genetic Algorithm?
I did not know what was TSP before starting this assignment so I started to read about the origin and the available solutions for TSP initially.
On the second day, when I was looking at some twitter blogs, this [article blogged by Eric Stoltz](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35) featured in Towards Data Science series came to my attention. The blog, "Evolution of a salesman: A complete genetic algorithm tutorial for Python", timely gave me a 'guidance' (when I was looking for an algorithm to implement) that my fate was developing a TSP solver based on Genetic Algorithm (GA). With this decision, I started to read some papers (references available at the bottom of this blog) and considered that given the time frame, I would try to customise many specific steps in GA by implementing some approaches from the papers.

## What is Genetic Algorithm?
GA is one of the simplest random-based evolutionary algorithms where the core concept stems from the Charles Darwin’s “survival of fittest” theory (Gad, 2018). Evolutionary algorithms are dynamic because they can “evolve” over time over “generations”. In a holistic picture, GA is based on cycles of four steps where each cycle/loop represents an evolution of a generation. The four steps are composed of:
* Initialisation;
* Selection;
* Crossover; and
* Mutation.

Gene in the TSP context is an individual city with its x and y coordinate. Chromosome is a solution consists of the list of genes, hence representing a path of the combined cities where every city is visited only once in the TSP context. Population is a set of chromosomes. In the initialisation step, this would be a set of paths generated from a list of cities for each problem. Individual city was implemented as a class suggested by Eric Stoltz's blog.

{% highlight  Python%}
class City:
	def __init__(self, node, x_coordinate, y_coordinate):
		self.node = int(node)
		self.x_coordinate = x_coordinate
		self.y_coordinate = y_coordinate

	def e2distance(self, another_city):
		x_length = self.x_coordinate - another_city.x_coordinate
		y_length = self.y_coordinate - another_city.y_coordinate
		e2d = math.sqrt(x_length*x_length + y_length*y_length)
		return e2d

	def to_string(self):
		return "city name: " + str(self.node) + " x coordinate: " + str(self.x_coordinate) + " y coordinate: " + str(self.y_coordinate)

	def __repr__(self):
		return self.to_string()

	def __str__(self):
		return self.to_string()
{% endhighlight %}

The key concept of the whole evolutions is 'fitness to survive'. Through reading some papers, this fitness is in general calculated as an inverse of total tour length of cities.
The shorter the total tour length is, the better the solution is. This means the fitness score is considered best where it is greater (because fitness is inverse of the tour length).
{% highlight  Python%}
def total_fitness(total_d):
	if total_d!=0.0:
		#make fitness inverse of total distance
		fitness = 10000000000000000.0/ total_d
	else:
		print("Total distance cannot be zero. Check again")
		sys.exit()
	return fitness
{% endhighlight %}

Based on the aforementioned concepts of gene, chromsome, and fitntess, the following pseudo algorithm is adopted from Dwivedi et al. (2012) to provide an overview of GA.

{% highlight  Python%}
Start
	Create initial population of various paths
	Define and calculate total distance and fitness of each path
	Start loop
	Selection by fitness criteria. Tournament selection if the number of cities is less than 150. Otherwise, Rank-based roulette wheel selection
	Edge recombination crossover.
	Swap mutation.
	Add new path to the population.
	Re-calculate fitness of the population.
Stop loop
{% endhighlight %}

## GA steps

### Initialisation of population

Deng et al. (2015) posits that an initialisation strategy of chromosome and population is important for optimisation. This project implements random and nearest-neighbour initialisation for population.

{% highlight  Python%}
#initialise with nearest neighbour
def create_a_path_n(cities):
    city = random.sample(cities,1)[0]
    path = [city]
    remaining_cities = [rc for rc in cities if rc!=city]
    #loop while the list of remaining cities are not empty
    while remaining_cities:
        #get the minimum distance
        city = min(remaining_cities, key=lambda c: c.e2distance(city))
        path.append(city)
        remaining_cities.remove(city)
    return path

#initialise randomly
def create_a_path(cities):
    #return unique elements for cities by using random.sample
    path = random.sample(cities, len(cities))
    return path

#create paths(population) of desired size - half random and half nearest neighbour
def create_paths(cities, n_path):
    paths = []
    point = int(n_path/2)
    for i in range(0, point):
        paths.append(create_a_path_n(cities))
    for i in range(point, n_path):
        paths.append(create_a_path(cities))
    return paths
{% endhighlight %}

### Selection strategies

Razali and Geraghty (2011) noted that selection is one of the important process in GA, experimenting different selection strategies to gauge performance. As a result of the tests, tournament selection strategy was considered to produce best solution quality for small size problems with low computing times than roulette wheel-based selection strategies.  However, because of more randomness in this strategy, convergence becomes slower as the size grows. In addition, if the size increases larger, it was found that tournament selection tends to resort to premature convergence. To alleviate this, rank-based roulette wheel selection is used for larger sized problems where paths are assigned with a linear rank function rather than with a proportion of a fitness score. Rank-based roulette wheel selection prevents premature convergence but is considered to be computationally-expensive. Therefore, this project implemented tournament selection strategy for small size TSP and rank-based roulette wheel selection strategy for large size TSP. 

{% highlight  Python%}
#for a small size TSP, use tournament strategy
def tournament_selection(tournament_sz, unsorted_paths, elite_sz):
    #best paths from sortedPaths are preserved for the size of elite
    sorted_paths = sort_paths(unsorted_paths)
    selected_paths = sorted_paths[:elite_sz]

    #remaining population is filled with tournament selection
    for i in range(0, len(sorted_paths)-elite_sz):
        #select unique random paths from sortedPaths
        in_tournament = random.sample(sorted_paths, tournament_sz)
        selected_paths.append(in_tournament[0])
    selected_paths = sort_paths(selected_paths)
    return selected_paths

#for a large size TSP, use rank-based roulette wheel
def rank_roulette_wheel_selection(unsorted_paths, elite_sz):
    #calculate each rank and total rank
    key_rank = cal_rank(unsorted_paths)
    total_rank = sum(rv for _, rv in key_rank)
    cum_key_rank = cal_cum_rank(key_rank)

    #best paths from sortedPaths are preserved for the size of elite
    sorted_paths = sort_paths(unsorted_paths)
    selected_paths = sorted_paths[:elite_sz]

    #remaining population is filled with rank-based roulette wheel selection
    while len(selected_paths) < len(sorted_paths):
        roulette_random = random.uniform(0.0, 100.0)
        for i in range(0, len(sorted_paths)):
            percent = cum_key_rank[i]/total_rank*100
            if percent >= roulette_random:
                key_path = list(cum_key_rank)[i]
                selected_paths.append(sorted_paths[key_path])
            else:
                key_path = list(cum_key_rank)[0]
                selected_paths.append(sorted_paths[key_path])
            if len(selected_paths) == len(sorted_paths):
                break
    selected_paths = sort_paths(selected_paths)
    return selected_paths
{% endhighlight %}

### Crossover

This project implemented two crossover methods: simple and edge recombination, which were combined together to create new paths. In simple crossover, two points in the first selected path are determined randomly, passing between-points cities to a new path. Any cities missing from a new path is then filled from the second selected path. Edge recombination, informed by Liu (2014)’s edgy swapping crossover, is implemented based on [pseudo algorithm listed on webpage](https://en.wikipedia.org/wiki/Edge_recombination_operator). On step 1, after selecting two existing paths similar to simple crossover, edges of each path is collated. On step 2, a union set is performed to get a unique adjacency matrix. On step 3, initialise the first city from a random parent. Most importantly, on step 4, create a new path in a loop by adding the city with the fewest neighbours or randomly selecting the city if there is no neighbour.

{% highlight  Python%}
###Edge recombination crossover
#build an edge list
def add_edges(path):
    path_edges = []
    #build edge route
    for i in range(0, len(path)):
        if i == 0:
            path_edges.append([path[i], (path[-1], path[i+1])])
        elif i == len(path)-1:
            path_edges.append([path[i], (path[i-1], path[0])])
        else:
            path_edges.append([path[i], (path[i-1], path[i+1])])
    return path_edges

#union two paths
def union_two_paths(path1_edges, path2_edges):
    union_edges = []
    path1_edges = sorted(path1_edges, key=lambda x: x[0].node, reverse=False)
    path2_edges = sorted(path2_edges, key=lambda x: x[0].node, reverse=False)
    for i in range(len(path1_edges)):
        union_edges.append([path1_edges[i][0], list(set().union(path1_edges[i][1], path2_edges[i][1]))])
    return union_edges

#calculate next neighbour from union_edges given origin
def get_nxt_neighbour(neighbours):
    #get the number of neighbours for each edge
    len_neighbours = []
    for edge in neighbours:
        len_neighbours.append((len(edge[1]), edge))
    #sort neighbours by the number of neighbours in each edge
    len_neighbours = sorted(len_neighbours, key=lambda x: x[0], reverse=False)
    #get edge with the smallest number of neighbours, if multiple, append them all
    nxt_neighbours = []
    for edge in len_neighbours:
        if edge[0]==len_neighbours[0][0]:
            nxt_neighbours.append(edge[1])
    #if there are multiple edges with the same number of neighbours, select a random edge
    if len(nxt_neighbours[0][1])>1:
        nxt_neighbours = random.sample(nxt_neighbours[0][1], 1)
    else:
        nxt_neighbours = nxt_neighbours[0][1]
    return nxt_neighbours[0]

#random neighbour
def get_rnd_neighbour(union_edges, new_path):
    #select a random edge from the remaining union_edges
    nxt_neighbour = random.sample(union_edges, 1)
    #while the selected random edge is in new_path, then reselect
    while nxt_neighbour[0] in new_path:
        nxt_neighbour = random.sample(union_edges[0][1], 1)
    return nxt_neighbour[0][0]

#edge recombination crossover to create a new path from selected original paths
def crossover_er(path1, path2):
    #https: // en.wikipedia.org / wiki / Edge_recombination_operator
    #step 1 get edges of each path
    path1_edges = add_edges(path1)
    path2_edges = add_edges(path2)
    #step 2 make a union to get unique adjacency matrix
    union_edges = union_two_paths(path1_edges, path2_edges)
    #step 3 initiate new_path and first city from a random parent
    new_path = []
    origin = random.choice([path1[0], path2[0]])
    #step 4 create a new path in a loop
    while len(new_path) < len(path1):
        #append the edge to a new path
        if origin not in new_path:
            new_path.append(origin)
        #stop appending if new_path has a full list of cities
        if len(new_path)==len(path1):
            break
        #get neighbour edge of origin
        neighbours = [edge for edge in union_edges if edge[0]==origin]
        #remove origin from all neighbour list
        for edge in union_edges:
            if edge[0].node == origin.node:
                union_edges.remove(edge)
        for edge in union_edges:
            for neighbour in edge[1]:
                if neighbour==origin:
                    edge[1].remove(neighbour)
        #if neighbours are not empty, let origin be the city with the fewest neighbours
        if len(neighbours[0][1])>0:
            nxt = get_nxt_neighbour(neighbours)
        #if not, origin be a random city that is not in a new_path
        else:
            nxt = get_rnd_neighbour(union_edges, new_path)
        #make nxt edge to origin edge and restart the loop
        origin = nxt
    return new_path
{% endhighlight %}

### Mutation

Mutation introduces diversity into paths. In this project, simple swap mutation is implemented.

{% highlight  Python%}
#for diversity, swap cities between in the path
def swap_cities(path):
    mutation_criteria = 0.6
    for original_index in range(1,len(path)):
        if mutation_criteria > random.uniform(0.0, 1.0):
            swapped_index = random.randint(0, len(path)-1)
            original_value = path[original_index]
            path[original_index] = path[swapped_index]
            path[swapped_index] = original_value
    return path

#do a swap mutation for all selected_paths except elites
def swap_cities_in_path(selected_paths, elite_sz):
    swapped_paths = []
    point = int(elite_sz)
    for index in range(0, point):
        swapped_paths = selected_paths[:point]
    for index in range(point, len(selected_paths)):
        swapped_path = swap_cities(selected_paths[index])
        swapped_paths.append(swapped_path)
    return swapped_paths
{% endhighlight %}

These were main four steps of the GA process. After the mutation step, the whole evolution process is to be looped through within a specified time frame.

## Query to obtain the best result for each TSP

As part of the assignment requirements, the solutions generated were stored in the database.
From the database, the query below is to retrieve the solution with a minimum distance for a particular TSP.

{% highlight  SQL%}
def get_best_solution(problem_name):
    connect = connect_db()
    cs = connect.cursor()
    #select ProblemName is same with the parameter and look for the minimum total distance available
    sql_query = "SELECT ts.* " \
                "FROM Solution ts " \
                "JOIN ( " \
                "SELECT ProblemName, MIN(TourLength) AS min_dis " \
                "FROM Solution " \
                "GROUP BY ProblemName ) AS ts2 " \
                "ON ts.ProblemName = ts2.ProblemName AND ts.TourLength = ts2.min_dis " \
                "WHERE ts.ProblemName = %s"
    prob_name = (problem_name, )
    cs.execute(sql_query, prob_name)
    result = cs.fetchall()
    connect.close()
    return result
{% endhighlight %}

## Results

With the aforementioned specific implementation, a greater variety of TSPs was run with the parameters of:
1. Mixed initialisation strategy;
2. Crossover strategy composed of half simple and half edge recombination; 
3. Mutation threshold rate of 0.6; 
4. Elitism rate of 0.6; 
5. Population size of 100,000; and 
6. Time limit of 600 seconds. 

| TSP    | eli51.tsp (optimal: 426) | berlin52.tsp(optimal: 7542) | d493.tsp (optimal: 35002) | d1655.tsp (optimal: 62128) | usa13509.tsp (optimal: [19947008, 19982889])|
|------- |:------------------------:|:---------------------------:|:-------------------------:|:--------------------------:|:-------------------------------------------:|
| Result | 433  					 | 7,548 					   | 40,914 				   | 74,936 				    | 790,994									  |

I have compared some of these results with those of my classmates and the result from the Simulated Annealing approach implemented by one classmate outperformed this GA. 
Overall, this TSP solver produced a reasonable, but not the best, result for each problem.

## Lessons learnt

This assignment was quite fun to code and taught me a great lesson - programming is not just about coding to solve direct problems. Rather, there were many side aspects to consider
to integrate the solver, the database, and the GUI together. Initially, I did not like to work on the GUI, thinking too much to do in order to make a little component of the GUI.
However, while I was working on it, the thought process of "If I am a user, how would I behave in this particular situation to achieve my goal?" became a norm and I do think that this is a valuable point in developing a program.

### References

A. Sebő and J. Vygen, “Shorter Tours by Nicer Ears: 7/5-approximation for graphic TSP, 3/2 for the path version, and 4/3 for two-edge-connected subgraphs”, arXiv:1201.1870v3 [cs.DM], Mar. 2012.

B. Johnson, “Genetic Algorithms: The Travelling Salesman Problem”, on Medium, https://medium.com/@becmjo/genetic-algorithms-and-the-travelling-salesman-problem-d10d1daf96a1, accessed on 22nd July 2018.

E. Stortz, “Evolution of a salesman: A complete genetic algorithm tutorial for Python”, on Medium,  https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35, accessed on 22nd July 2018.

J. Yu, D. Yue, D, and F. You, Traveling salesman problems, https://optimization.mccormick.northwestern.edu/index.php/Traveling_salesman_problems, last modified on 26th May 2014, accessed on 21st July 2018.

M. Hahsler and K. Hornik, "TSP – Infrastructure for the Traveling Salesperson Problem", Journal of Statistical Software, vol. 23 issue. 2, p. 1–21, 2007.

N. M. Razali and J. Geraghty, “Genetic Algorithm Performance with Different Selection Strategies in Solving TSP”, Proceedings of the World Congress on Engineering, Jul. 2011.

R. Matai, S. Singh, and M. Lal, “Traveling salesman problem: An overview of applications, formulations, and solution approaches”, In D. Davendra (Ed.), Traveling Salesman Problem, Theory and Applications, InTech, 2010.

V. Dwivedi, T. Chauhan, S. Saxena, and P. Agrawal, “Travelling Salesman Problem using Genetic Algorithm”, National Conference on Development of Reliable Information Systems, Techniques and Related Issues, 2012.

Y. Deng, Y Liu, and D. Zhou, “An Improved Genetic Algorithm with Initial Population Strategy for Symmetric TSP,” Mathematical Problems in Engineering, vol. 2015, Article ID 212794, 6 pages, 2015. https://doi.org/10.1155/2015/212794, accessed on 22nd July 2018.