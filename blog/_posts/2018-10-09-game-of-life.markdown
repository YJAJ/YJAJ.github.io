---
layout: post
author: YJ Park
title:  "Conway's Game of Life (assignment with Java)"
date:   2018-10-09 20:01:59 +1000
categories: jekyll update
tags: Conway's Game of Life, Game of Life, Java
---
This assignment was the fourth project of 1801ICT (Programming languages). The requirements included that
* A project should be written in Java;
* Inheritance and polymorphism should be incorporated in a project; and
* A project is to be preferably interfaced with GUI.

I was looking around examples of small projects that I could try in a time limit of four or five days (because I was preparing for Viva Voce for my PhD).
This list on quora named ["Pro/g/ramming Challenges v1.4e"](https://www.quora.com/What-are-some-small-projects-I-could-do-using-Java) helped me to search for an interesting project.
I was tempted by Minesweeper, however, decided to try Conway's Game of Life since the assignment from my other course also uses a theme of "evolution through generations" (i.e. Genetic Algorithm).

Conway's Game of Life ([Wikipedia Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) describes the origin and simple rules of the implementation. It is noted that the Game of Life is a cellular finite automaton developed by John Horton Conway in 1970. It has four simple rules as follows:
* Any live cell with two or three live neighbours lives on to the next generation.
* Any live cell with fewer than two live neighbours dies, as if by under population.
* Any live cell with more than three live neighbours dies, as if by overpopulation.
* Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

With these rules, cells will live or die over generation. For example, this Pulsar evolution is quite pretty. 

![Wikipedia Game of Life pulsar](https://upload.wikimedia.org/wikipedia/commons/0/07/Game_of_life_pulsar.gif)

source: https://upload.wikimedia.org/wikipedia/commons/0/07/Game_of_life_pulsar.gif

As I had no idea around GUI for Java, I had to read many posts from Stack Overflow implementing different ways of Game of Life. The important part here was that user input of mouse click should be the initialisation of the first generation. After reading around some posts, I decided to divide the implementation into three four different classes. They are
* Cell class that has the information about x coordinate, y coordinate, and the state of alive or dead with associated getter and setter methods;
* Game board class that imitates the grid cells through embedded JPanels in a large JPanel;
* Runner class that possesses the rules of Game of Life and returns the next evolved generation; and
* Main and GUI class that has user interface elements.

The constructor of Cell class has the information of x coordinate, y coordinate and cell size (in this implementation, it is fixed as 20). Cell class inherited from JPanel class.
{% highlight  Java%}
Cell(int x, int y, int cellSize)
    {
        xPos = x;
        yPos = y;
        alive = 0;
        setBackground(new Color(0, 102, 204));
        setPreferredSize(new Dimension(cellSize, cellSize));
    }
{% endhighlight %}

Most important setter methods of Cell class include toggling between the alive and dead state and setting the state of a particular cell based on the argument.
The codes below do not have public or private in front of type because they are package private. I use IntelliJ and this IDE suggests lots of better pratices like these ones.
{% highlight  Java%}
    void toggleAlive()
    {
        if (alive==0)
        {
            alive=1;
        }
        else
        {
            alive=0;
        }
    }

    void setAlive(int aliveOrNot)
    {
        alive = aliveOrNot;
    }
{% endhighlight %}

Gameboard class, which inherited from JPanel class, has only a constructor method. It instantiates a Gameboard object that imitates many cells in a square grid.
I used JPanel for a large Gameboard and then with x and y coordinates, added small JPanels within a large Gameboard.
By setting a background colour of a large Gameboard different from a background colour of little JPanels, Gameboard objects appear to be grid cells (suggested on Stack Overflow).
{% highlight  Java%}
class GameBoard extends JPanel {

    private static final int gap = 1;
    private static final Color bg = new Color(0, 102, 204);
    private static final int cellSize = 20;

    GameBoard(int[][] generation, int side)
    {
        JPanel [][] placeHolder = new JPanel[side][side];
        setPreferredSize(new Dimension(22*side, 22*side));
        setBackground(bg);
        setLayout(new GridLayout(side,side,gap,gap));
        for (int i=0; i < side; ++i)
        {
            for (int j=0; j < side; ++j)
            {
                placeHolder[i][j] = new JPanel();
                placeHolder[i][j].setBackground(Color.black);
                add(placeHolder[i][j]);
                final Cell cell = new Cell(i, j, cellSize);
                cell.setAlive(generation[i][j]);
                if (cell.getAlive()==1)
                {
                    placeHolder[i][j].setBackground(new Color(204, 204, 0));
                }
            }
        }
    }
}
{% endhighlight %}

Runner class is simple. It just implements the four rules. The first rule that if a cell is alive and two or three neighbours are also alive, then the cell will live on is implicit in the implementation.

{% highlight  Java%}
    private static int gameOfLifeRules(int aliveNeighbours, int alive)
    {
        //Any live cell with two or three live neighbors lives on to the next generation where it is already alive.
        int aliveOrNot = alive;
        //Any live cell with fewer than two live neighbors dies, as if by under population.
        if (aliveNeighbours < 2 && alive == 1)
        {
            aliveOrNot = 0;
        }
        //Any live cell with more than three live neighbors dies, as if by overpopulation.
        else if (aliveNeighbours > 3 && alive == 1)
        {
            aliveOrNot = 0;
        }
        //Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
        else if (aliveNeighbours == 3 && alive == 0)
        {
            aliveOrNot = 1;
        }
        return aliveOrNot;
    }
{% endhighlight %}

Finally, ConwayGameOfLife class has all important interface. I implemented the first version of Gameboard here without using Gameboard class because the initialisation should take in user input.
The effect of mouse clicked or unclicked is expressed with the coulour change (clicked - purple, unclicked - original black). The constructor of this class implements this.
The constructor has a method to call the setter of Cell objects - toggleAlive() - that sets the state of alive or dead cell. 

{% highlight  Java%}
    private ConwayGameOfLife(int side)
    {
        //create original generation with the size of sides
        JPanel [][] placeHolder = new JPanel[side][side];
        setPreferredSize(new Dimension(22*side, 22*side));
        passedGeneration = new int[side][side];
        setBackground(bg);
        setLayout(new GridLayout(side, side, gap, gap));

        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++)
            {
                placeHolder[i][j] = new JPanel();
                //placeHolder[i][j].setOpaque(true);
                placeHolder[i][j].setBackground(Color.black);
                placeHolder[i][j].setBorder(BorderFactory.createLineBorder(Color.black));
                add(placeHolder[i][j]);

                int cellSize = 20;
                final Cell cell = new Cell(i, j, cellSize);
                placeHolder[i][j].addMouseListener(new MouseAdapter()
                {
                    public void mouseClicked(MouseEvent e)
                    {
                        if (cell.getAlive()==0)
                        {
                            int xPos = cell.getX();
                            int yPos = cell.getY();
                            placeHolder[xPos][yPos].setBackground(new Color(204, 204, 0));
                        }
                        else
                        {
                            int xPos = cell.getX();
                            int yPos = cell.getY();
                            placeHolder[xPos][yPos].setBackground(Color.black);
                        }
                        cell.toggleAlive();
                        createOriginalGeneration(passedGeneration, cell);
                    }
                });
            }
        }
    }
{% endhighlight %}

In addition, through createOriginalGeneration(), a copy of the state is then passed to create the first generation that would be eventually passed to Runner class to go through the rules.

{% highlight  Java%}
    private void createOriginalGeneration(int[][] generation, Cell cell)
    {
        int xPosition = cell.getX();
        int yPosition = cell.getY();
        generation[xPosition][yPosition] = cell.getAlive();
    }
{% endhighlight %}

That's it! Here is the example of Game Of Life.
<video width="500" height="500" controls><source src="../../../../../../assets/videos/game_of_life_play.mp4" type="video/mp4"></video>


If you would like to try Game of Life, it can be found here ["Conway's Game of Life"](https://bitbucket.org/YJAJ/1801ict_project_4/src/master)