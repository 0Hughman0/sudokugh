import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

RC_MASK = np.repeat(True, 9)
POSSIBLES = set(range(1, 10))

vals = np.vectorize(lambda cell: cell.val)

class Cell:
    
    def __init__(self, grid, irow, icol):
        self.possibles = POSSIBLES.copy()
        self.grid = grid
        self.irow = irow
        self.icol = icol
        
        self._row_mask = RC_MASK.copy()
        self._col_mask = RC_MASK.copy()
        self._box_mask = RC_MASK.copy().reshape(3, 3)
        self._row_mask[self.icol] = False
        self._col_mask[self.irow] = False
        self._box_mask[self.irow - self.irbox, self.icol - self.icbox] = False
        
    @property
    def irbox(self):
        return self.irow - self.irow % 3
    
    @property
    def icbox(self):
        return self.icol - self.icol % 3
    
    @property
    def box(self):
        return self.grid[self.irbox:self.irbox+3, self.icbox:self.icbox+3][self._box_mask]
    
    @property
    def row(self):
        return self.grid[self.irow, :][self._row_mask]
    
    @property
    def col(self):
        return self.grid[:, self.icol][self._col_mask]
        
    @property
    def fixed(self):
        return len(self) <= 1
    
    @property
    def val(self):
        if self.fixed:
            return list(self.possibles)[0]
        else:
            return -1
    
    def fix(self, val):
        self.possibles = {val}
    
    def __len__(self):
        return self.possibles.__len__()
    
    def __repr__(self):
        return f"<Cell {self.irow, self.icol} {self.possibles}>"
    
    def __hash__(self):
        return (self.irow, self.icol, self.possibles).__hash__()
    
    

class Grid:

    def __init__(self):
        self.grid = np.array([[Cell(self, row, col) for col in range(9)] for row in range(9)])
        
    def __getitem__(self, irowcol):
        irow, icol = irowcol
        return self.grid[irow, icol]
    
    @classmethod
    def from_string(cls, grid_string):
        g = grid_string.strip()
        cleaned = [l.split(',') for l in g.replace('|', '').replace(' ', '0').split('\n')]
        arrayed = np.array(cleaned).astype(int)
        
        return cls.from_array(arrayed)
    
    @classmethod
    def from_input(cls):
        o = cls()
        for i in range(9):
            print("New row\n", i+1)
            for j in range(9):
                if j % 3:
                    print("New Box\n")

                v = input(f"Enter value for cell {i + 1, j + 1}, blank for no value")
                if v:
                    o[i, j].fix(int(v))
        
        return o
    
    @classmethod
    def from_array(cls, array):
        """
        0 or -1 for blank squares!
        """
        o = cls()
        
        for row, grow in zip(array, o.grid):
            for num, cell in zip(row, grow):
                if num in POSSIBLES:
                    cell.fix(num)
        
        return o
    
    def as_array(self):
        return vals(self.grid)
    
    def copy(self):
        return self.from_array(self.as_array())
    
    @property
    def broken(self):
        return any(len(cell.possibles) <= 0 for cell in self.grid.flatten())
                        
    def ifixed(self, collection=None):
        if collection is None:
            collection = self.grid

        for cell in collection.flatten():
            if cell.fixed:
                yield cell
    
    def iunfixed(self, collection=None):
        if collection is None:
            collection = self.grid

        for cell in collection.flatten():
            if not cell.fixed:
                yield cell
                
    def find_required(self, collection):
        required = POSSIBLES.copy()
        for fixed in self.ifixed(collection):
            required -= fixed.possibles
        return required
                
    def count_fixed(self, collection=None):
        if collection is None:
            collection = self.grid
        return sum(1 for cell in self.ifixed(collection))
    
    def nearest_complete(self):
        all_unfixed = list(self.iunfixed())
        all_unfixed.sort(key=lambda cell: (len(cell.possibles), # smallest number of candidates
                         (8 * 3) - sum(self.count_fixed(collection) for collection in (cell.row, cell.col, cell.box))))  # largest amount of additional info!
        return all_unfixed
            
    def gen_targets(self):
        while not all(cell.fixed for cell in self.grid.flatten()):
            nearest_complete = self.nearest_complete()
    
            count = self.count_fixed()
            
            for cell in nearest_complete:
                yield cell
                
                if self.count_fixed() != count:  # New fixed cells, need to start again!
                    break
            else:
                raise RuntimeError("Hit roadblock, all cells presented")  # got to the end and made no progress
            
    
    def display(self):
        array = self.as_array()
        return np.where(array != -1, array, ' ').astype(str)
    
    def find_possibles(self, cell):
        eliminated = set()
        for collection in (cell.row, cell.col, cell.box):
            for other in self.ifixed(collection):
                eliminated.update(other.possibles)
        return POSSIBLES ^ eliminated 
    
    def update_possibles(self):
        for unfixed in self.iunfixed():
            unfixed.possibles = self.find_possibles(unfixed)
        
        if self.broken:
            raise RuntimeError("Impossible to solve cell found")
    
    def try_elimination(self, unfixed):
        for collection in (unfixed.row, unfixed.col, unfixed.box):
            possibles = self.find_required(collection) & unfixed.possibles
            
            alternatives = self.iunfixed(collection)
            
            for alternative in alternatives:
                possibles -= alternative.possibles
            
            if len(possibles) == 1:    
                unfixed.possibles = possibles
                return True
        return False
                
    def solve(self):
        logging.info("Performing initial reduction")
        i = 0
        self.update_possibles()
        logging.info("Eliminating possibilities")
        
        for i, target in enumerate(self.gen_targets()):
            logging.info(f"Trying to deduce {target}")
            if self.try_elimination(target):
                logging.info(f"Success! {target.val}")
                logging.info("Updating consequences")
                self.update_possibles()
            else:
                logging.info("No luck!")
        
        logging.info("Solved")
        return i + 1

    def deepsolve(self, pmin=1/9 ** 3, _p=1.0):
        try:
            i = self.solve()
            logging.info("Deep solved")
            return i
        
        except RuntimeError:
            logging.info("Got stuck, attempting to branch")
            
            nearest_complete = self.nearest_complete()
            
            for iba, candidate in enumerate(nearest_complete):
            
                _p = (1 / len(candidate.possibles) * _p)
                
                if _p < pmin:
                    raise RuntimeError("Chance of success judged too low")
                
                for possible in candidate.possibles:
                    branch_grid = self.copy()
                    branch_grid[candidate.irow, candidate.icol].fix(possible)
                    
                    try:
                        logging.info(f"Entering a branch for {candidate} using {possible}")
                        complete = branch_grid.deepsolve(pmin=pmin, _p=_p)
                        self.from_array(complete.as_array())
                        return complete
                    except Exception as e:
                        logging.info(f"Deadend reached due to {e}")
                        del branch_grid
                        

if __name__ == '__main__':
    from tests import T1, T2, T3, T4
    g1 = Grid.from_array(T1)
    g2 = Grid.from_array(T2)
    g3 = Grid.from_array(T3)
    g4 = Grid.from_array(T4)
    