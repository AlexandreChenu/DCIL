import random
import math
import time
from .cell import Cell
import numpy as np
import pylab as pl
from matplotlib import collections  as mc


class Maze(object):
    """Class representing a maze; a 2D grid of Cell objects. Contains functions
    for generating randomly generating the maze as well as for solving the maze.

    Attributes:
        num_cols (int): The height of the maze, in Cells
        num_rows (int): The width of the maze, in Cells
        id (int): A unique identifier for the maze
        grid_size (int): The area of the maze, also the total number of Cells in the maze
        entry_coor Entry location cell of maze
        exit_coor Exit location cell of maze
        generation_path : The path that was taken when generating the maze
        solution_path : The path that was taken by a solver when solving the maze
        initial_grid (list):
        grid (list): A copy of initial_grid (possible this is un-needed)
        """

    def __init__(self, num_rows, num_cols, id=0, seed=None, standard=False):
        """Creates a gird of Cell objects that are neighbors to each other.

            Args:
                    num_rows (int): The width of the maze, in cells
                    num_cols (int): The height of the maze in cells
                    id (id): An unique identifier

        """
        print("MAZE setting random seed ",seed)
        random.seed(seed)
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.id = id
        self.grid_size = num_rows*num_cols
        self.entry_coor = self._pick_random_entry_exit(None)
        self.exit_coor = self._pick_random_entry_exit(self.entry_coor)
        self.generation_path = []
        self.solution_path = None
        self.initial_grid = self.generate_grid()
        self.grid = self.initial_grid
        if not standard:
            self.generate_maze((0, 0))
        elif self.num_cols == 3:
            self.grid[0][0].remove_walls(1,0)
            self.grid[1][0].remove_walls(0,0)

            self.grid[1][0].remove_walls(2,0)
            self.grid[2][0].remove_walls(1,0)

            self.grid[2][0].remove_walls(2,1)
            self.grid[2][1].remove_walls(2,0)

            self.grid[2][1].remove_walls(1,1)
            self.grid[1][1].remove_walls(2,1)

            self.grid[1][1].remove_walls(0,1)
            self.grid[0][1].remove_walls(1,1)

            self.grid[0][1].remove_walls(0,2)
            self.grid[0][2].remove_walls(0,1)

            self.grid[0][2].remove_walls(1,2)
            self.grid[1][2].remove_walls(0,2)

            self.grid[1][2].remove_walls(2,2)
            self.grid[2][2].remove_walls(1,2)
        elif self.num_cols == 2:
            self.grid[0][0].remove_walls(1,0)
            self.grid[1][0].remove_walls(0,0)

            self.grid[1][0].remove_walls(1,1)
            self.grid[1][1].remove_walls(1,0)

            self.grid[1][1].remove_walls(0,1)
            self.grid[0][1].remove_walls(1,1)
        elif self.num_cols == 4:
            self.empty_grid()
            for i in range(self.num_rows):
                if i != 0:
                    self.grid[i][1].add_walls(i,2)
                    self.grid[i][2].add_walls(i,1)
        elif self.num_cols == 8:
            self.empty_grid()
            for i in range(self.num_rows):
                if i != 1:
                    self.grid[i][3].add_walls(i,4)
                    self.grid[i][4].add_walls(i,3)
        else:
            print("ERROR : No standard maze for size ",self.num_cols)

    def empty_grid(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i!=0:
                    self.grid[i][j].remove_walls(i-1,j)
                if j!=0:
                    self.grid[i][j].remove_walls(i,j-1)
                if i!=self.num_rows-1:
                    self.grid[i][j].remove_walls(i+1,j)
                if j!=self.num_cols-1:
                    self.grid[i][j].remove_walls(i,j+1)

    def no_grid(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                    self.grid[i][j].remove_walls(i-1,j)
                    self.grid[i][j].remove_walls(i,j-1)
                    self.grid[i][j].remove_walls(i+1,j)
                    self.grid[i][j].remove_walls(i,j+1)

    def generate_grid(self):
        """Function that creates a 2D grid of Cell objects. This can be thought of as a
        maze without any paths carved out

        Return:
            A list with Cell objects at each position

        """

        # Create an empty list
        grid = list()

        # Place a Cell object at each location in the grid
        for i in range(self.num_rows):
            grid.append(list())

            for j in range(self.num_cols):
                grid[i].append(Cell(i, j))

        return grid

    def find_neighbours(self, cell_row, cell_col):
        """Finds all existing and unvisited neighbours of a cell in the
        grid. Return a list of tuples containing indices for the unvisited neighbours.

        Args:
            cell_row (int):
            cell_col (int):

        Return:
            None: If there are no unvisited neighbors
            list: A list of neighbors that have not been visited
        """
        neighbours = list()

        def check_neighbour(row, col):
            # Check that a neighbour exists and that it's not visited before.
            if row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols:
                neighbours.append((row, col))

        check_neighbour(cell_row-1, cell_col)     # Top neighbour
        check_neighbour(cell_row, cell_col+1)     # Right neighbour
        check_neighbour(cell_row+1, cell_col)     # Bottom neighbour
        check_neighbour(cell_row, cell_col-1)     # Left neighbour

        if len(neighbours) > 0:
            return neighbours

        else:
            return None     # None if no unvisited neighbours found

    def _validate_neighbours_generate(self, neighbour_indices):
        """Function that validates whether a neighbour is unvisited or not. When generating
        the maze, we only want to move to move to unvisited cells (unless we are backtracking).

        Args:
            neighbour_indices:

        Return:
            True: If the neighbor has been visited
            False: If the neighbor has not been visited

        """

        neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def validate_neighbours_solve(self, neighbour_indices, k, l, k_end, l_end, method = "fancy"):
        """Function that validates whether a neighbour is unvisited or not and discards the
        neighbours that are inaccessible due to walls between them and the current cell. The
        function implements two methods for choosing next cell; one is 'brute-force' where one
        of the neighbours are chosen randomly. The other is 'fancy' where the next cell is chosen
        based on which neighbour that gives the shortest distance to the final destination.

        Args:
            neighbour_indices
            k
            l
            k_end
            l_end
            method

        Return:


        """
        if method == "fancy":
            neigh_list = list()
            min_dist_to_target = 100000

            for k_n, l_n in neighbour_indices:
                if (not self.grid[k_n][l_n].visited
                        and not self.grid[k][l].is_walls_between(self.grid[k_n][l_n])):
                    dist_to_target = math.sqrt((k_n - k_end) ** 2 + (l_n - l_end) ** 2)

                    if (dist_to_target < min_dist_to_target):
                        min_dist_to_target = dist_to_target
                        min_neigh = (k_n, l_n)

            if "min_neigh" in locals():
                neigh_list.append(min_neigh)

        elif method == "brute-force":
            neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited
                          and not self.grid[k][l].is_walls_between(self.grid[n[0]][n[1]])]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def _pick_random_entry_exit(self, used_entry_exit=None):
        """Function that picks random coordinates along the maze boundary to represent either
        the entry or exit point of the maze. Makes sure they are not at the same place.

        Args:
            used_entry_exit

        Return:

        """
        rng_entry_exit = used_entry_exit    # Initialize with used value

        # Try until unused location along boundary is found.
        while rng_entry_exit == used_entry_exit:
            rng_side = random.randint(0, 3)

            if (rng_side == 0):     # Top side
                rng_entry_exit = (0, random.randint(0, self.num_cols-1))

            elif (rng_side == 2):   # Right side
                rng_entry_exit = (self.num_rows-1, random.randint(0, self.num_cols-1))

            elif (rng_side == 1):   # Bottom side
                rng_entry_exit = (random.randint(0, self.num_rows-1), self.num_cols-1)

            elif (rng_side == 3):   # Left side
                rng_entry_exit = (random.randint(0, self.num_rows-1), 0)

        return rng_entry_exit       # Return entry/exit that is different from exit/entry

    def generate_maze(self, start_coor = (0, 0)):
        """This takes the internal grid object and removes walls between cells using the
        depth-first recursive backtracker algorithm.

        Args:
            start_coor: The starting point for the algorithm

        """
        k_curr, l_curr = start_coor             # Where to start generating
        path = [(k_curr, l_curr)]               # To track path of solution
        self.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visit_counter = 1                       # To count number of visited cells
        visited_cells = list()                  # Stack of visited cells for backtracking

        #print("\nGenerating the maze with depth-first search...")
        # time_start = time.clock()

        while visit_counter < self.grid_size:     # While there are unvisited cells
            neighbour_indices = self.find_neighbours(k_curr, l_curr)    # Find neighbour indicies
            neighbour_indices = self._validate_neighbours_generate(neighbour_indices)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                k_next, l_next = random.choice(neighbour_indices)     # Choose random neighbour
                self.grid[k_curr][l_curr].remove_walls(k_next, l_next)   # Remove walls between neighbours
                self.grid[k_next][l_next].remove_walls(k_curr, l_curr)   # Remove walls between neighbours
                self.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path
                visit_counter += 1

            elif len(visited_cells) > 0:  # If there are no unvisited neighbour cells
                k_curr, l_curr = visited_cells.pop()      # Pop previous visited cell (backtracking)
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path

        #print("Number of moves performed: {}".format(len(path)))
        #print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))

        self.grid[self.entry_coor[0]][self.entry_coor[1]].set_as_entry_exit("entry",
            self.num_rows-1, self.num_cols-1)
        self.grid[self.exit_coor[0]][self.exit_coor[1]].set_as_entry_exit("exit",
            self.num_rows-1, self.num_cols-1)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.grid[i][j].visited = False      # Set all cells to unvisited before returning grid

        self.generation_path = path

    def __str__(self):
        buffer = [[] for i in range(len(self.grid)*2+1)]
        for i in range(len(self.grid)*2+1):
            buffer[i] = [(
                "+" if i%2==0 and j%2==0 else
                "-" if i%2==0 and j%2==1 else
                "|" if i%2==1 and j%2==0 else
                " ") for j in range(len(self.grid[0])*2+1)]
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if not self.grid[i][j].walls["top"]:
                    buffer[i*2+1-1][j*2+1]=" "
                if not self.grid[i][j].walls["bottom"]:
                    buffer[i*2+1+1][j*2+1]=" "
                if not self.grid[i][j].walls["left"]:
                    buffer[i*2+1][j*2+1-1]=" "
                if not self.grid[i][j].walls["right"]:
                    buffer[i*2+1][j*2+1+1]=" "
        s=""
        for r in buffer:
            for c in r:
                s += c
            s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

    def draw(self,ax,thick=0.0):
        lines = []
        if thick>0:
            def add_hwall(lines,i,j):
                t=thick
                lines.append([(i-t,j),(i-t,j+1)])
                lines.append([(i-t,j+1),(i+t,j+1)])
                lines.append([(i+t,j),(i+t,j+1)])
                lines.append([(i+t,j),(i-t,j)])
            def add_vwall(lines,i,j):
                t=thick
                lines.append([(i,j-t),(i+1,j-t)])
                lines.append([(i+1,j-t),(i+1,j+t)])
                lines.append([(i,j+t),(i+1,j+t)])
                lines.append([(i,j+t),(i,j-t)])
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    if self.grid[i][j].walls["top"]:
                        add_hwall(lines,i,j)
                    if self.grid[i][j].walls["bottom"]:
                        add_hwall(lines,i+1,j)
                    if self.grid[i][j].walls["left"]:
                        add_vwall(lines,i,j)
                        lines.append([(i,j),(i+1,j)])
                    if self.grid[i][j].walls["right"]:
                        add_vwall(lines,i,j+1)
        else:
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    if self.grid[i][j].walls["top"]:
                        lines.append([(i,j),(i,j+1)])
                    if self.grid[i][j].walls["bottom"]:
                        lines.append([(i+1,j),(i+1,j+1)])
                    if self.grid[i][j].walls["left"]:
                        lines.append([(i,j),(i+1,j)])
                    if self.grid[i][j].walls["right"]:
                        lines.append([(i,j+1),(i+1,j+1)])
        lc = mc.LineCollection(lines, linewidths=2)

        ax.add_collection(lc)
