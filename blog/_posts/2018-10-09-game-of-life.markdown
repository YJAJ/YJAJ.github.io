---
layout: post
author: YJ Park
title:  "Conway's Game of Life (assignment with Java)"
date:   2018-10-09 20:01:59 +1000
categories: jekyll update
---
This assignment was the fourth project of 1801ICT (Programming languages). The requirements included that
* A project should be written in Java;
* Inheritance and polymorphism should be incorporated in a project; and
* A project is to be preferably interfaced with GUI.

I was looking around examples of small projects that I could try in a time limit of four or five days (because I was preparing for Viva Voce for my PhD).
This list on quora named ["Pro/g/ramming Challenges v1.4e"](https://www.quora.com/What-are-some-small-projects-I-could-do-using-Java) helped me to search for an interesting project.
I was tempted by Minesweeper, however, decided to try Conway's Game of Life since the assignment from my other course also uses a theme of "evolution through generations" (i.e. Genetic Algorithm).

Conway's Game of Life ([Wikipedia Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) describes the origin and simple rules of the implementatin.

As I had no ideas around GUI for Java, I had to read many posts from Stack Overflow implementing different ways of Game of Life. The important part here was that user input of mouse click should be the initialisation of the first generation.


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

If you would like to try Game of Life, it can be found here ["Conway's Game of Life"](https://bitbucket.org/YJAJ/1801ict_project_4/src/master )